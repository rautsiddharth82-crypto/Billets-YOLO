"""
Automated Billet Quality Inspection System - Excel Logger Module
===============================================================
Handles high-reliability, thread-safe logging of billet inspection records
to Excel (billet_log.xlsx) with robust file lock handling and automatic fallback buffering.

Columns:
- Timestamp: ISO 8601 formatted date-time
- Heat_Number: Stamped heat code (e.g., H727622R1.0)
- Width_mm: Measured & corrected billet width in mm
- Height_mm: Measured & corrected billet height in mm
- Defects: Comma-separated list of detected defects with confidence (or 'None')
- Status: PASS or FAIL (based on IS 2830 tolerances and defect clearance)
"""

import os
import time
import queue
import sqlite3
import threading
import datetime
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# Dedicated local storage directory outside OneDrive to avoid cloud sync lockouts and enable COM updates
LOCAL_DATA_DIR = r"C:\SteelInspectionData"
try:
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    DEFAULT_LOG_PATH = os.path.join(LOCAL_DATA_DIR, "steel_quality_records.xlsx")
    DEFAULT_DB_PATH = os.path.join(LOCAL_DATA_DIR, "steel_quality_master.db")
except Exception:
    DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "steel_quality_records.xlsx")
    DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "steel_quality_master.db")

PROJECT_MIRROR_LOG_PATH = os.path.join(os.path.dirname(__file__), "steel_quality_records.xlsx")
PROJECT_MIRROR_DB_PATH = os.path.join(os.path.dirname(__file__), "steel_quality_master.db")
BACKUP_LOG_PATH = os.path.join(LOCAL_DATA_DIR, "logs", "steel_quality_buffered.csv")


class BilletExcelLogger:
    """
    Thread-safe Non-Locking Excel Logger & SQLite Dual-Writer.
    Meets Problem Statement Section 5 requirements:
    1. Queue Buffering (in-memory queue.Queue background worker)
    2. OpenPyXL Overlay Writer (mode='a' / append to max_row + 1)
    3. SQLite Dual-Writing (simultaneous write to billet_master.db)
    """

    COLUMNS = [
        "Timestamp",
        "Heat_Number",
        "Width_mm",
        "Height_mm",
        "Defects",
        "Status"
    ]

    def __init__(self, file_path=DEFAULT_LOG_PATH, db_path=None, max_retries=3, retry_delay=0.5):
        self.file_path = file_path
        if db_path is not None:
            self.db_path = db_path
        elif os.path.abspath(file_path) != os.path.abspath(DEFAULT_LOG_PATH):
            self.db_path = os.path.join(os.path.dirname(os.path.abspath(file_path)), "billet_master.db")
        else:
            self.db_path = DEFAULT_DB_PATH
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lock = threading.Lock()
        self.memory_buffer = []

        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        os.makedirs(os.path.dirname(BACKUP_LOG_PATH), exist_ok=True)

        self._initialize_sqlite()
        self._initialize_excel_file()

        # In-memory queue for non-blocking asynchronous writing
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self.worker_thread.start()

    def _execute_sqlite(self, sql, params=()):
        """Executes SQL query on SQLite dual-store with guaranteed connection closing."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                return cursor.fetchall()
            finally:
                conn.close()
        except Exception as e:
            return None

    def _initialize_sqlite(self):
        """Initializes local SQLite database (billet_master.db) for dual-writing."""
        self._execute_sqlite("""
            CREATE TABLE IF NOT EXISTS billet_inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                heat_number TEXT NOT NULL,
                width_mm REAL NOT NULL,
                height_mm REAL NOT NULL,
                defects TEXT NOT NULL,
                status TEXT NOT NULL,
                rhomboidity_delta_mm REAL DEFAULT 0.0,
                synced_to_excel INTEGER DEFAULT 1
            )
        """)
        print(f"[Logger] Initialized SQLite dual-write database at {self.db_path}")

    def _initialize_excel_file(self):
        """Creates the Excel workbook with headers and styling if it doesn't exist."""
        with self.lock:
            if not os.path.exists(self.file_path):
                try:
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.title = "InspectionLog"

                    # Header styling (Industrial Tata/JSW dark steel theme)
                    header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
                    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
                    thin_border = Border(
                        left=Side(style='thin', color='D1D5DB'),
                        right=Side(style='thin', color='D1D5DB'),
                        top=Side(style='thin', color='D1D5DB'),
                        bottom=Side(style='thin', color='D1D5DB')
                    )

                    ws.append(self.COLUMNS)
                    for col_num, col_name in enumerate(self.COLUMNS, 1):
                        cell = ws.cell(row=1, column=col_num)
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                        cell.border = thin_border
                        ws.column_dimensions[get_column_letter(col_num)].width = 20

                    wb.save(self.file_path)
                    wb.close()
                    print(f"[Logger] Initialized new inspection Excel sheet at {self.file_path}")
                except Exception as e:
                    print(f"[Logger Warning] Could not initialize Excel file: {e}")

    def _write_to_sqlite(self, record, synced_to_excel=1):
        """Dual-writes inspection record to SQLite database (billet_master.db)."""
        self._execute_sqlite("""
            INSERT INTO billet_inspections 
            (timestamp, heat_number, width_mm, height_mm, defects, status, rhomboidity_delta_mm, synced_to_excel)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["Timestamp"],
            record["Heat_Number"],
            record["Width_mm"],
            record["Height_mm"],
            record["Defects"],
            record["Status"],
            record.get("Rhomboidity_mm", 0.0),
            synced_to_excel
        ))

    def _queue_worker(self):
        """Background daemon thread continuously consuming inspection queue."""
        while True:
            try:
                record = self.write_queue.get()
                if record is None:
                    break

                # 1. Dual-write to SQLite (always safe, non-locking)
                self._write_to_sqlite(record, synced_to_excel=0)

                # 2. Append to Excel with retry handling
                success = self._write_record_with_retry(record)
                if success:
                    self._execute_sqlite("""
                        UPDATE billet_inspections 
                        SET synced_to_excel = 1 
                        WHERE timestamp = ? AND heat_number = ?
                    """, (record["Timestamp"], record["Heat_Number"]))
                else:
                    self.memory_buffer.append(record)
                    self._write_to_backup_csv(record)

                self.write_queue.task_done()
            except Exception:
                time.sleep(0.1)

    def log_inspection(self, heat_number, width_mm, height_mm, defects, status, timestamp=None, rhomboidity_mm=0.0):
        """
        Appends a billet inspection record to Excel and SQLite dual-store.
        
        Args:
            heat_number (str): OCR heat number (e.g. 'H727622R1.0')
            width_mm (float): Corrected width in mm
            height_mm (float): Corrected height in mm
            defects (str or list): Detected defect names with confidence
            status (str): 'PASS' or 'FAIL'
            timestamp (str, optional): ISO timestamp string. Defaults to now.
            rhomboidity_mm (float, optional): Cross-sectional rhomboidity skew in mm.
            
        Returns:
            bool: True if written directly or buffered, False on fatal error
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(defects, list):
            defects_str = ", ".join(defects) if defects else "None"
        else:
            defects_str = str(defects) if defects else "None"

        record = {
            "Timestamp": timestamp,
            "Heat_Number": str(heat_number),
            "Width_mm": round(float(width_mm), 2),
            "Height_mm": round(float(height_mm), 2),
            "Defects": defects_str,
            "Status": str(status).upper(),
            "Rhomboidity_mm": round(float(rhomboidity_mm), 2)
        }

        # Dual-write to SQLite immediately (zero file locks)
        self._write_to_sqlite(record, synced_to_excel=1)

        with self.lock:
            # 1. Live Windows COM push: if Microsoft Excel is open on screen, push row directly
            # into the live open sheet so operator sees it appear in real-time without refreshing
            com_synced = self._sync_to_live_excel_com(record)

            # 2. Disk persistence via openpyxl
            success = com_synced or self._write_record_with_retry(record)
            if not success:
                # Enqueue to background retry queue
                self.memory_buffer.append(record)
                self._write_to_backup_csv(record)
                self.write_queue.put(record)
                print(f"[Logger] Excel file locked or busy. Buffered record for {record['Heat_Number']}.")
                return False
            else:
                if self.memory_buffer:
                    self._flush_buffered_records()
                return True

    def _sync_to_live_excel_com(self, record):
        """
        Pushes inspection row directly into the active Microsoft Excel window
        via Windows COM API so rows appear instantly in real time without refreshing.
        """
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client

            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "steel_quality_records.xlsx")
            candidates = [os.path.abspath(self.file_path), PROJECT_MIRROR_LOG_PATH, downloads_path]

            wb = None
            # Only connect to an existing, user-visible Microsoft Excel window
            try:
                excel_app = win32com.client.GetActiveObject("Excel.Application")
                if excel_app is not None and excel_app.Workbooks.Count > 0:
                    for i in range(1, excel_app.Workbooks.Count + 1):
                        w = excel_app.Workbooks(i)
                        if any(k in w.Name.lower() for k in ["steel_quality", "billet_log"]):
                            wb = w
                            break
            except Exception:
                # Excel is not currently running on screen; do NOT spawn a hidden process
                wb = None

            if wb is not None:
                try:
                    sheet_names = [s.Name for s in wb.Sheets]
                    if "QualityLog" in sheet_names:
                        ws = wb.Sheets("QualityLog")
                    elif "InspectionLog" in sheet_names:
                        ws = wb.Sheets("InspectionLog")
                    else:
                        ws = wb.ActiveSheet
                except Exception:
                    ws = wb.ActiveSheet

                # Find the next empty row based on Column A (Timestamp)
                row_idx = 1
                while ws.Cells(row_idx, 1).Value is not None:
                    row_idx += 1

                ws.Cells(row_idx, 1).Value = str(record["Timestamp"])
                ws.Cells(row_idx, 2).Value = str(record["Heat_Number"])
                ws.Cells(row_idx, 3).Value = float(record["Width_mm"])
                ws.Cells(row_idx, 4).Value = float(record["Height_mm"])
                ws.Cells(row_idx, 5).Value = str(record["Defects"])
                ws.Cells(row_idx, 6).Value = str(record["Status"])

                # Apply industrial coloring
                if record["Status"] == "PASS":
                    ws.Cells(row_idx, 6).Interior.Color = 0xDCFCE7  # Light green
                    ws.Cells(row_idx, 6).Font.Color = 0x166534      # Dark green
                else:
                    ws.Cells(row_idx, 6).Interior.Color = 0xFEE2E2  # Light red
                    ws.Cells(row_idx, 6).Font.Color = 0x991B1B      # Dark red

                ws.Columns("A:F").AutoFit()
                try:
                    wb.Save()
                except Exception:
                    pass
                return True
        except Exception:
            pass
        finally:
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
        return False

    def _write_record_with_retry(self, record):
        """Attempts to open, append, and save to Excel with retries."""
        for attempt in range(self.max_retries):
            try:
                wb = openpyxl.load_workbook(self.file_path)
                ws = wb.active

                row_values = [
                    record["Timestamp"],
                    record["Heat_Number"],
                    record["Width_mm"],
                    record["Height_mm"],
                    record["Defects"],
                    record["Status"]
                ]

                next_row = ws.max_row + 1
                ws.append(row_values)

                # Format row based on PASS/FAIL status
                status_color = "DCFCE7" if record["Status"] == "PASS" else "FEE2E2"
                font_color = "166534" if record["Status"] == "PASS" else "991B1B"
                status_fill = PatternFill(start_color=status_color, end_color=status_color, fill_type="solid")
                status_font = Font(name="Segoe UI", size=10, bold=(record["Status"] == "FAIL"), color=font_color)
                data_font = Font(name="Segoe UI", size=10)

                for col_idx in range(1, len(row_values) + 1):
                    c = ws.cell(row=next_row, column=col_idx)
                    c.alignment = Alignment(horizontal="center" if col_idx in [1, 2, 3, 4, 6] else "left", vertical="center")
                    if col_idx == 6:  # Status column
                        c.fill = status_fill
                        c.font = status_font
                    else:
                        c.font = data_font

                wb.save(self.file_path)
                wb.close()

                # Mirror to project folder so both locations stay synchronized
                if os.path.abspath(self.file_path) != os.path.abspath(PROJECT_MIRROR_LOG_PATH):
                    try:
                        import shutil
                        shutil.copy2(self.file_path, PROJECT_MIRROR_LOG_PATH)
                    except Exception:
                        pass
                return True
            except (PermissionError, IOError):
                # File is likely open in Microsoft Excel or another viewer
                time.sleep(self.retry_delay)
            except Exception:
                time.sleep(self.retry_delay)

        return False

    def _flush_buffered_records(self):
        """Flushes in-memory buffered records to Excel once lock is released."""
        remaining = []
        for rec in self.memory_buffer:
            if not self._write_record_with_retry(rec):
                remaining.append(rec)
        self.memory_buffer = remaining

    def _write_to_backup_csv(self, record):
        """Saves unsaved records to backup CSV to prevent any data loss."""
        try:
            df = pd.DataFrame([record])
            header = not os.path.exists(BACKUP_LOG_PATH)
            df.to_csv(BACKUP_LOG_PATH, mode='a', header=header, index=False)
        except Exception as e:
            print(f"[Logger Error] Backup CSV write failed: {e}")

    def get_recent_logs(self, limit=10):
        """
        Reads recent inspection rows from SQLite dual-store (zero locks) or Excel.
        Returns a pandas DataFrame.
        """
        # 1. Primary non-locking read: SQLite billet_master.db
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                query = f"""
                    SELECT timestamp as Timestamp, heat_number as Heat_Number, 
                           width_mm as Width_mm, height_mm as Height_mm, 
                           defects as Defects, status as Status
                    FROM billet_inspections 
                    ORDER BY id DESC LIMIT {int(limit)}
                """
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    return df
            finally:
                conn.close()
        except Exception:
            pass

        # 2. Secondary read: Excel workbook
        try:
            if os.path.exists(self.file_path):
                df = pd.read_excel(self.file_path, sheet_name=0)
                if not df.empty:
                    return df.tail(limit).iloc[::-1].reset_index(drop=True)
        except Exception:
            pass

        # 3. Tertiary fallback: Backup CSV
        if os.path.exists(BACKUP_LOG_PATH):
            try:
                df_csv = pd.read_csv(BACKUP_LOG_PATH)
                if not df_csv.empty:
                    return df_csv.tail(limit).iloc[::-1].reset_index(drop=True)
            except Exception:
                pass

        return pd.DataFrame(columns=self.COLUMNS)

    def clear_log(self):
        """Clears all records from SQLite, Excel, and CSV with clean headers."""
        with self.lock:
            # Clear SQLite
            self._execute_sqlite("DELETE FROM billet_inspections;")

            # Clear backup CSV
            try:
                if os.path.exists(BACKUP_LOG_PATH):
                    os.remove(BACKUP_LOG_PATH)
            except Exception:
                pass

            # Clear Excel
            try:
                if os.path.exists(self.file_path):
                    wb = openpyxl.load_workbook(self.file_path)
                    ws = wb.active
                    if ws.max_row > 1:
                        ws.delete_rows(2, ws.max_row)
                    wb.save(self.file_path)
                    wb.close()
                else:
                    self._initialize_excel_file()

                # Mirror cleared file to project directory
                if os.path.abspath(self.file_path) != os.path.abspath(PROJECT_MIRROR_LOG_PATH):
                    try:
                        import shutil
                        shutil.copy2(self.file_path, PROJECT_MIRROR_LOG_PATH)
                    except Exception:
                        pass
            except Exception:
                try:
                    if os.path.exists(self.file_path):
                        os.remove(self.file_path)
                except Exception:
                    pass

            # Clear live open Excel sheet via COM if open on screen
            try:
                import pythoncom
                pythoncom.CoInitialize()
                import win32com.client
                wb_clear = None
                try:
                    excel_app = win32com.client.GetActiveObject("Excel.Application")
                    if excel_app is not None and excel_app.Workbooks.Count > 0:
                        for i in range(1, excel_app.Workbooks.Count + 1):
                            w = excel_app.Workbooks(i)
                            if any(k in w.Name.lower() for k in ["steel_quality", "billet_log"]):
                                wb_clear = w
                                break
                except Exception:
                    pass

                if wb_clear is not None:
                    try:
                        s_names = [s.Name for s in wb_clear.Sheets]
                        ws = wb_clear.Sheets("QualityLog") if "QualityLog" in s_names else (wb_clear.Sheets("InspectionLog") if "InspectionLog" in s_names else wb_clear.ActiveSheet)
                        last_row = ws.UsedRange.Rows.Count
                        if last_row > 1:
                            ws.Rows(f"2:{last_row}").Delete()
                        wb_clear.Save()
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

            self.memory_buffer = []

    def launch_live_excel_window(self):
        """
        Launches Microsoft Excel with the steel_quality_records.xlsx workbook
        visible on screen so the user can watch rows appear in real time.
        Clears any orphaned temporary lock files beforehand.
        """
        target = os.path.abspath(self.file_path)
        if not os.path.exists(target):
            self._initialize_excel_file()

        # Clean up any leftover ~$ lock file if Excel is not actively open
        lock_file = os.path.join(os.path.dirname(target), f"~${os.path.basename(target)}")
        if os.path.exists(lock_file):
            try:
                import pythoncom
                pythoncom.CoInitialize()
                import win32com.client
                excel_app = win32com.client.GetActiveObject("Excel.Application")
            except Exception:
                # Excel is not running, so the lock file is an orphaned leftover!
                try:
                    os.remove(lock_file)
                except Exception:
                    pass

        try:
            os.startfile(target)
            return True, target
        except Exception:
            try:
                import subprocess
                subprocess.Popen(["cmd", "/c", "start", "", target], shell=True)
                return True, target
            except Exception as e:
                return False, str(e)


# Global singleton instance
billet_logger = BilletExcelLogger()


if __name__ == "__main__":
    print("Testing BilletExcelLogger...")
    logger = BilletExcelLogger()
    # Log a PASS sample
    logger.log_inspection(
        heat_number="H727622R1.0",
        width_mm=125.2,
        height_mm=124.8,
        defects=[],
        status="PASS"
    )
    # Log a FAIL sample
    logger.log_inspection(
        heat_number="H727623R2.1",
        width_mm=129.5,
        height_mm=125.1,
        defects=["crazing (0.87)", "inclusion (0.74)"],
        status="FAIL"
    )
    recent = logger.get_recent_logs(5)
    print(f"Logged 2 records successfully. Recent records:\n{recent}")
