from concurrent.futures import CancelledError
import os
import threading
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
from llm_search.indexer import Indexer
from llm_search.utils import safe_str_to_int
from llm_search.searcher import Searcher
import pystray
import subprocess
import logging

ICON_PATH = 'icon.ico'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexer = Indexer()
indexer.start_indexer_thread()
searcher = Searcher(indexer_ref=indexer)

def on_clicked(icon, item):
    app_window.deiconify()
    app_window.state('normal')

def setup(icon):
    icon.visible = True

def clear_results():
    app_window.result_tree.delete(*app_window.result_tree.get_children())

def update_results(results):
    app_window.result_tree.delete(*app_window.result_tree.get_children())
    for i, result in enumerate(results):
        file_path, similarity = result
        filename = os.path.basename(file_path)
        app_window.result_tree.insert("", "end", values=(filename, file_path, f"{similarity:.2f}"))

current_future = None  
def on_search_input_change(*args):
    global current_future
    query = app_window.search_var.get().strip()
    if query:
        if current_future and not current_future.done():
            current_future.cancel()

        current_future = searcher.search(query, top_n=safe_str_to_int(app_window.top_n_var.get()))
        current_future.add_done_callback(handle_future_result)
    else:
        clear_results()

indexer.set_index_callback(on_search_input_change)

def handle_future_result(future):
    try:
        results = future.result()
        app_window.after(0, update_results, results)
    except CancelledError:
        logger.info("Search was canceled.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def minimize_to_tray():
    app_window.withdraw()
    icon.visible = True

def open_file(event):
    selected_item = app_window.result_tree.selection()[0]
    file_path = app_window.result_tree.item(selected_item)['values'][1]
    try:
        os.startfile(file_path)
    except:
        subprocess.run(['xdg-open', file_path])  # For Linux systems

def load_icon_image(icon_path, size=(32, 32)):
    with Image.open(icon_path) as img:
        img = img.resize(size, Image.LANCZOS)
        return img

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LLM Search')
        self.geometry('1000x700')
        self.minsize(800, 600)
        
        self.configure(bg='#1E1E1E')
        self.font = font.nametofont("TkDefaultFont")
        self.font.configure(size=14)
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#1E1E1E')
        self.style.configure('TLabel', background='#1E1E1E', foreground='#FFFFFF')
        self.style.configure('TEntry', fieldbackground='#2E2E2E', foreground='#FFFFFF', borderwidth=0)
        self.style.configure('Treeview', background='#2E2E2E', fieldbackground='#2E2E2E', foreground='#FFFFFF', borderwidth=0, rowheight=25)
        self.style.configure('Treeview.Heading', background='#3E3E3E', foreground='#FFFFFF', borderwidth=0)
        self.style.configure('TSeparator', background='#3E3E3E')
        self.style.configure('Vertical.TScrollbar', gripcount=0,
                            background='#2E2E2E', darkcolor='#1E1E1E', lightcolor='#3E3E3E',
                            troughcolor='#2E2E2E', bordercolor='#1E1E1E', arrowcolor='#FFFFFF')
        
        self.create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", minimize_to_tray)
        
        # Set the taskbar icon
        self.set_app_icon()

    def set_app_icon(self):
        icon_path = os.path.abspath(ICON_PATH)
        img = Image.open(icon_path)
        img = img.resize((32, 32), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.iconphoto(False, photo)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="30 30 30 30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 20))
        
        search_label = ttk.Label(search_frame, text="Search:", font=('TkDefaultFont', 16))
        search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", on_search_input_change)
        
        search_box = ttk.Entry(search_frame, textvariable=self.search_var, font=('TkDefaultFont', 16))
        search_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        top_n_label = ttk.Label(search_frame, text="Top N:", font=('TkDefaultFont', 16))
        top_n_label.pack(side=tk.LEFT, padx=(10, 10))
        
        self.top_n_var = tk.StringVar(value='10')
        self.top_n_var.trace_add("write", on_search_input_change)
        top_n_picker = ttk.Combobox(search_frame, textvariable=self.top_n_var, font=('TkDefaultFont', 16), state='readonly')
        top_n_picker['values'] = ('5', '10', '20', '50', 'all')
        top_n_picker.pack(side=tk.LEFT)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=20)
        
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_tree = ttk.Treeview(result_frame, columns=("Filename", "Path", "Similarity"), show="headings", style="Treeview")
        self.result_tree.heading("Filename", text="Filename")
        self.result_tree.heading("Path", text="Path")
        self.result_tree.heading("Similarity", text="Similarity")
        
        self.result_tree.column("Filename", width=200, anchor=tk.W)
        self.result_tree.column("Path", width=500, anchor=tk.W)
        self.result_tree.column("Similarity", width=100, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview, style='Vertical.TScrollbar')
        self.result_tree.configure(yscroll=scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_tree.bind("<Double-1>", open_file)

    def on_closing(self):
        self.withdraw()
        icon.stop()
        self.destroy()

app_window = App()

def on_quit(icon, item):
    searcher.shutdown()
    indexer.stop_indexer_thread()
    icon.stop()
    app_window.destroy()

icon_image = load_icon_image(ICON_PATH, size=(32, 32))
icon = pystray.Icon('name', icon_image, 'LLM Search', menu=pystray.Menu(
    pystray.MenuItem('Open', on_clicked),
    pystray.MenuItem('Quit', on_quit)
))

icon_thread = threading.Thread(target=icon.run, daemon=True)
icon_thread.start()

try:
    app_window.mainloop()
except Exception as e:
    logger.error(f"An error occurred in the main loop: {e}")
finally:
    searcher.shutdown()
    indexer.stop_indexer_thread()
    icon.stop()
