import os
import threading
import time
from queue import Queue
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import pickle
from llm_search.utils import try_cast_int
import pystray
import faiss
import numpy as np
import sqlite3
import subprocess
import logging
from llm_search.config import Config
from llm_search.embeddings import get_document_embeddings, load_model_and_tokenizer
from llm_search.extractor import extract_text_from_file, is_text_file
from llm_search.database import create_database, insert_or_update_embedding, get_last_modified_time, load_embeddings
from llm_search.search import search
import sys
import win32gui
import win32con
import win32api

ICON_PATH = 'icon.ico'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and set up FAISS index
_, _, _, device = load_model_and_tokenizer()
logger.info(f"Using device: {device}")
dimension = get_document_embeddings("test for dimension").shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
project_db_path = Config.DB_PATH
create_database(project_db_path)

stop_event = threading.Event()  

def initialize_faiss_index():
    conn = sqlite3.connect(project_db_path)
    c = conn.cursor()
    c.execute("SELECT id, embedding FROM embeddings ORDER BY id")
    results = c.fetchall()
    conn.close()
    if results:
        embeddings = []
        ids = []
        for result in results:
            id_, embedding_blob = result
            embedding = np.array(pickle.loads(embedding_blob))
            embeddings.append(embedding)
            ids.append(id_)
        embeddings = np.asarray(embeddings)
        ids = np.asarray(ids)
        index.add_with_ids(embeddings, ids)

initialize_faiss_index()

def on_clicked(icon, item):
    app_window.deiconify()
    app_window.state('normal')

def on_quit(icon, item):
    stop_event.set()
    icon.stop()
    app_window.quit()

def setup(icon):
    icon.visible = True

def update_index(directory, force_update=False):
    logger.info(f"Updating index for directory: {directory}...")
    eligible_files = []
    for root, _, files in os.walk(directory):
        if stop_event.is_set():
            break
        for file in files:
            if stop_event.is_set():
                break
            file_path = os.path.join(root, file)
            if is_text_file(file_path):
                eligible_files.append(file_path)
   
    for file_path in eligible_files:
        if stop_event.is_set():
            break
        last_modified = os.path.getmtime(file_path)
        db_last_modified = get_last_modified_time(project_db_path, file_path)
        if force_update or db_last_modified is None or last_modified > db_last_modified:
            text = extract_text_from_file(file_path)
            if text and text != "":
                embeddings = get_document_embeddings(text)
                chunk_ids = insert_or_update_embedding(project_db_path, file_path, embeddings, last_modified)
                
                # Update FAISS index
                # Get all existing chunk indices
                conn = sqlite3.connect(project_db_path)
                c = conn.cursor()
                c.execute('SELECT id, chunk_index FROM embeddings WHERE file_path = ?', (file_path,))
                existing_chunks = c.fetchall()
                conn.close()

                existing_chunk_ids = set([chunk_id for chunk_id, _ in existing_chunks])
                new_chunk_ids = set(chunk_ids)

                # Remove old embeddings that are no longer present
                chunks_to_remove = existing_chunk_ids - new_chunk_ids
                if chunks_to_remove:
                    index.remove_ids(np.array(list(chunks_to_remove)))

                # Add new embeddings
                for chunk_index, embedding in enumerate(embeddings):
                    chunk_id = chunk_ids[chunk_index]
                    index.add_with_ids(np.array([embedding]), np.array([chunk_id]))

    logger.info("Index update completed.")

search_queue = Queue()
search_thread = None
current_search_id = 0

def clear_results():
    app_window.result_tree.delete(*app_window.result_tree.get_children())

def search_worker():
    global current_search_id
    while True:
        if stop_event.is_set():
            break
        search_id, query = search_queue.get()
        if search_id != current_search_id:
            search_queue.task_done()
            continue

        top_n = try_cast_int(app_window.top_n_var.get())
        results = search(index, query, project_db_path, top_n=top_n)
        if search_id == current_search_id:
            app_window.after(0, update_results, results)
        search_queue.task_done()

def update_results(results):
    app_window.result_tree.delete(*app_window.result_tree.get_children())
    for i, result in enumerate(results):
        file_path, similarity = result
        filename = os.path.basename(file_path)
        app_window.result_tree.insert("", "end", values=(filename, file_path, f"{similarity:.2f}"))

def on_search_input_change(*args):
    global current_search_id
    query = app_window.search_var.get().strip()
    if query:
        current_search_id += 1
        search_queue.put((current_search_id, query))
    else:
        app_window.result_tree.delete(*app_window.result_tree.get_children())

def start_search_thread():
    global search_thread
    search_thread = threading.Thread(target=search_worker, daemon=True)
    search_thread.start()

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
        
        if sys.platform.startswith('win'):
            # For Windows
            self.iconbitmap(default=icon_path)
            
            # Set taskbar icon
            myappid = 'mycompany.myproduct.subproduct.version'  # arbitrary string
            win32gui.LoadImage(win32gui.GetModuleHandle(None), icon_path, win32con.IMAGE_ICON, 0, 0, win32con.LR_LOADFROMFILE)
            win32api.SetConsoleTitle(myappid)
        else:
            # For non-Windows platforms
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

app_window = App()

icon_image = load_icon_image(ICON_PATH, size=(32, 32))
icon = pystray.Icon('name', icon_image, 'LLM Search', menu=pystray.Menu(
    pystray.MenuItem('Open', on_clicked),
    pystray.MenuItem('Quit', on_quit)
))

def background_indexing():
    while True:
        if stop_event.is_set():
            break
        logger.info("Starting background indexing...")
        for directory in Config.INDEX_DIRECTORIES:
            if stop_event.is_set():
                break
            update_index(directory, force_update=False)
        time.sleep(Config.RE_INDEX_INTERVAL_SECONDS)

index_thread = threading.Thread(target=background_indexing, daemon=True)
index_thread.start()

threading.Thread(target=icon.run, daemon=True).start()
start_search_thread()
app_window.mainloop()