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
from llm_search.config import Config

ICON_PATH = 'icon.ico'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

indexer = Indexer()
indexer.start_indexer()
searcher = Searcher(indexer_ref=indexer)

def on_clicked(icon, item):
    app_window.deiconify()
    app_window.state('normal')

def setup(icon):
    icon.visible = True

def clear_results():
    app_window.result_tree.delete(*app_window.result_tree.get_children())

def update_results(results):
    existing_items = app_window.result_tree.get_children()
    
    for i, result in enumerate(results):
        file_path, similarity = result
        filename = os.path.basename(file_path)
        values = (filename, file_path, f"{similarity:.2f}")
        
        if i < len(existing_items):
            app_window.result_tree.item(existing_items[i], values=values)
        else:
            app_window.result_tree.insert("", "end", values=values)
    
    # Remove extra items if there are fewer results than existing items
    for item in existing_items[len(results):]:
        app_window.result_tree.delete(item)

current_future = None
def on_search_input_change(*args):
    global current_future
    query = app_window.search_var.get().strip()
    
    if not query:
        clear_results()
        return

    if current_future is not None and not current_future.done():
        current_future.cancel()

    try:
        new_future = searcher.search(query, top_n=safe_str_to_int(app_window.top_n_var.get()))
        if new_future is not None:
            current_future = new_future
            current_future.add_done_callback(handle_future_result)
        else:
            logger.warning("Search did not return a valid future.")
            clear_results()
    except Exception as e:
        logger.error(f"Error during search: {e}")

indexer.set_index_callback(on_search_input_change)

def handle_future_result(future):
    try:
        # Handle if the future has been cancelled
        if future.cancelled():
            logger.info("Search was canceled.")
            return
        
        # Get the results from the future
        results = future.result()
        if results is not None:
            app_window.after(0, update_results, results)
        else:
            logger.warning("Search returned None results.")
            clear_results()
    except CancelledError:
        logger.info("Future was canceled.")
    except Exception as e:
        logger.error(f"An error occurred during search: {e}")
        clear_results()

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

import threading

update_thread = None
update_button = None

def force_update():
    global update_thread, update_button
    
    def update_task():
        indexer.update_indexes(Config.INDEX_DIRECTORIES)
        app_window.search_var.set(app_window.search_var.get())  # Trigger a new search
        update_button.config(state='normal')  # Re-enable the button
        
    if update_thread is None or not update_thread.is_alive():
        update_thread = threading.Thread(target=update_task)
        update_thread.start()
        update_button.config(state='disabled')  # Disable the button while updating

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('LLM Search')
        self.geometry('1000x700')
        self.minsize(800, 600)
        
        self.configure(bg='#2C2F33')
        self.font = font.nametofont("TkDefaultFont")
        self.font.configure(size=12, family="Segoe UI")
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2C2F33')
        self.style.configure('TLabel', background='#2C2F33', foreground='#FFFFFF', font=('Segoe UI', 12))
        self.style.configure('TEntry', fieldbackground='#23272A', foreground='#FFFFFF', borderwidth=0, font=('Segoe UI', 12))
        self.style.configure('Treeview', background='#23272A', fieldbackground='#23272A', foreground='#FFFFFF', borderwidth=0, rowheight=30, font=('Segoe UI', 11))
        self.style.configure('Treeview.Heading', background='#7289DA', foreground='#FFFFFF', borderwidth=1, font=('Segoe UI', 12, 'bold'))
        self.style.configure('TSeparator', background='#7289DA')
        self.style.configure('Vertical.TScrollbar', gripcount=0,
                            background='#23272A', darkcolor='#2C2F33', lightcolor='#7289DA',
                            troughcolor='#23272A', bordercolor='#2C2F33', arrowcolor='#FFFFFF')
        self.style.configure('TButton', background='#7289DA', foreground='#FFFFFF', borderwidth=0, font=('Segoe UI', 12, 'bold'), padding=5)
        self.style.map('TButton', background=[('active', '#677BC4')])
        
        self.create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", minimize_to_tray)
        
        # Set the taskbar icon
        self.set_app_icon()


    def create_widgets(self):
        global update_button
        main_frame = ttk.Frame(self, padding="30 30 30 30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 20))
        
        search_label = ttk.Label(search_frame, text="Search:", font=('Segoe UI', 14, 'bold'))
        search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", on_search_input_change)
        
        search_box = ttk.Entry(search_frame, textvariable=self.search_var, font=('Segoe UI', 14))
        search_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        top_n_label = ttk.Label(search_frame, text="Top N:", font=('Segoe UI', 14, 'bold'))
        top_n_label.pack(side=tk.LEFT, padx=(20, 10))
        
        self.top_n_var = tk.StringVar(value='10')
        self.top_n_var.trace_add("write", on_search_input_change)
        top_n_picker = ttk.Combobox(search_frame, textvariable=self.top_n_var, font=('Segoe UI', 14), state='readonly', width=5)
        top_n_picker['values'] = ('5', '10', '20', '50', 'all')
        top_n_picker.pack(side=tk.LEFT)
        
        update_button = ttk.Button(search_frame, text="Force Update", command=force_update, style='TButton')
        update_button.pack(side=tk.RIGHT, padx=(20, 0))
        
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
        
    def set_app_icon(self):
        icon_path = os.path.abspath(ICON_PATH)
        img = Image.open(icon_path)
        img = img.resize((32, 32), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.iconphoto(False, photo)

    def on_closing(self):
        self.withdraw()
        icon.stop()
        self.destroy()

app_window = App()

def on_quit(icon, item):
    searcher.shutdown()
    indexer.stop_indexer()
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
    indexer.stop_indexer()
    icon.stop()