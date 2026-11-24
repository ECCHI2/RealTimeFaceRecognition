"""Vize: ortak yuz tespiti baslaticisi (Windows / Tkinter)."""
import os
import re
import tempfile
import ctypes
from ctypes import wintypes
import uuid
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
MODELS = {
    'Haar Cascade': ('face_detection', '.venv'),
    'MediaPipe': ('mediapipe', '.venv'),
    'YOLOv8n-Face': ('yolo', '.venv_yolo'),
}
SOURCES = {'Resim': 'image', 'Kamera': 'camera', 'Video': 'video'}


# Windows Job Object: the worker joins before loading any model.
# Closing the last job handle kills its members, even if the GUI is killed.
class JobBasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD)]


class JobIOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
        'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class JobExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', JobBasicLimits),
                ('IoInfo', JobIOCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]


class DetectionJob:
    def __init__(self):
        self.name = 'Local\\FaceDetection_' + uuid.uuid4().hex
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.handle = self.api.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = JobExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


# Run inside the selected Python environment. Assign the real Python worker
# (not just the Windows venv redirector) to the job before starting detection.
WORKER_BOOTSTRAP = r"""
import ctypes, sys, runpy
from ctypes import wintypes
from pathlib import Path
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
k.OpenJobObjectW.restype = wintypes.HANDLE
k.GetCurrentProcess.restype = wintypes.HANDLE
k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
k.AssignProcessToJobObject.restype = wintypes.BOOL
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.CloseHandle.restype = wintypes.BOOL
job_name, script, stop_path, source_kind, selected_file = sys.argv[1:6]
h = k.OpenJobObjectW(0x0001, False, job_name)  # JOB_OBJECT_ASSIGN_PROCESS
if not h:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    if not k.AssignProcessToJobObject(h, k.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
finally:
    k.CloseHandle(h)
# All existing scripts use OpenCV windows and waitKey. Route close/stop to Q.
import cv2
# Redirect only the first file input of the selected image/video script.
# No test files are overwritten and no camera/model paths are changed.
used_input = False
if selected_file:
    original_imread = cv2.imread
    original_capture = cv2.VideoCapture
    def selected_imread(filename, *args, **kwargs):
        global used_input
        if source_kind == 'image' and not used_input:
            used_input = True
            print('Seçilen resim:', selected_file, flush=True)
            # imdecode + fromfile also supports Unicode paths on Windows.
            import numpy as np
            flags = args[0] if args else kwargs.get('flags', cv2.IMREAD_COLOR)
            return cv2.imdecode(np.fromfile(selected_file, dtype=np.uint8), flags)
        return original_imread(filename, *args, **kwargs)
    def selected_capture(*args, **kwargs):
        global used_input
        filename = args[0] if args else kwargs.get('filename')
        if source_kind == 'video' and not used_input and isinstance(filename, (str, Path)):
            used_input = True
            print('Seçilen video:', selected_file, flush=True)
            if args:
                args = (selected_file,) + args[1:]
            else:
                kwargs['filename'] = selected_file
        return original_capture(*args, **kwargs)
    if source_kind == 'image':
        cv2.imread = selected_imread
    elif source_kind == 'video':
        cv2.VideoCapture = selected_capture
stop = Path(stop_path)
windows = set()
original_show = cv2.imshow
original_wait = cv2.waitKey

def show(name, image):
    original_show(name, image)
    windows.add(name)

def exit_requested():
    if stop.exists():
        return True
    for name in tuple(windows):
        try:
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
    return False

def wait(delay=0):
    if delay <= 0:
        while True:
            if exit_requested():
                return ord('q')
            key = original_wait(50)
            if key != -1:
                return key
    if exit_requested():
        return ord('q')
    key = original_wait(delay)
    return ord('q') if exit_requested() else key

cv2.imshow = show
cv2.waitKey = wait
sys.argv = [script] + sys.argv[6:]
sys.path.insert(0, str(Path(script).parent))
runpy.run_path(script, run_name='__main__')
if selected_file and not used_input:
    raise RuntimeError('Seçilen dosya kullanılmadı. Giriş yöntemini kontrol edin.')
"""


# The dashboard launches existing detection scripts without changing them.
BG = '#0b1220'
PANEL = '#131e30'
FIELD = '#1e2b40'
EDGE = '#2b3b52'
TEXT = '#f1f5fb'
MUTED = '#a4b5cc'
ACCENT = '#3bd6b0'
BLUE = '#7baaff'
RED = '#ff8995'
FONT = 'Segoe UI'


def device_choices(model, source):
    if model != 'YOLOv8n-Face':
        return ('CPU',)
    return ('GPU', 'CPU') if source == 'Video' else ('GPU',)


def parse_metrics(line):
    """Return only measurements actually emitted by a detection script."""
    line = line.strip()
    patterns = {
        'fps': r'(?:Detection FPS|Ortalama detection FPS|Ortalama FPS)\s*:\s*([\d.,]+)',
        'latency': r'(?:Ortalama tespit(?: suresi)?|Ortalama detection suresi|Ortalama tespit sure\w*|Toplam tespit suresi \(ms\))\s*(?:\(ms\))?\s*:\s*([\d.,]+)',
        'inference': r'(?:Ortalama inference|Inference suresi \(ms\))\s*:\s*([\d.,]+)',
        'frames': r'(?:Islenen frame|Toplam frame|Islenen toplam frame)\s*:\s*(\d+)',
        'faces': r'(?:Bulunan yuz sayisi|Toplam yuz tespiti)\s*:\s*(\d+)',
    }
    result = {}
    for key, pattern in patterns.items():
        m = re.search(pattern, line, re.I)
        if m:
            try:
                v = float(m.group(1).replace(',', '.'))
                result[key] = str(int(v)) if key in ('frames', 'faces') else f'{v:.2f}'
            except ValueError:
                pass
    return result


class DetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('FaceVision | Gerçek Zamanlı Yüz Tespiti')
        self.geometry('1120x790')
        self.minsize(1000, 735)
        self.configure(bg=BG)
        self.process = None
        self.job = None
        self.stop_path = None
        self.stopping = False
        self.forced = False
        self.closing = False
        self.messages = queue.Queue()
        self.model = tk.StringVar(value='Haar Cascade')
        self.source = tk.StringVar(value='Resim')
        self.device = tk.StringVar(value='CPU')
        self.selected_files = {'Resim': None, 'Video': None}
        self.status = tk.StringVar(value='Hazır')
        self.session = tk.StringVar(value='Henüz oturum başlatılmadı')
        self.environment = tk.StringVar()
        self.metrics = {k: tk.StringVar(value='—') for k in ('fps','latency','inference','frames','faces')}
        self.protocol('WM_DELETE_WINDOW', self.close_app)
        self.setup_style()
        self.build_ui()
        self.update_options()
        self.after(80, self.poll_messages)

    def setup_style(self):
        st = ttk.Style(self)
        st.theme_use('clam')
        st.configure('Modern.TCombobox', fieldbackground=FIELD, background=FIELD,
                     foreground=TEXT, arrowcolor=MUTED, bordercolor=EDGE,
                     lightcolor=EDGE, darkcolor=EDGE, padding=5,
                     selectbackground=FIELD, selectforeground=TEXT, font=(FONT, 11))
        st.map('Modern.TCombobox', fieldbackground=[('readonly', FIELD), ('disabled', PANEL)],
               foreground=[('disabled', MUTED), ('readonly', TEXT)],
               selectbackground=[('readonly', FIELD)], selectforeground=[('readonly', TEXT)])
        self.option_add('*TCombobox*Listbox.background', FIELD)
        self.option_add('*TCombobox*Listbox.foreground', TEXT)
        self.option_add('*TCombobox*Listbox.selectBackground', '#345577')
        self.option_add('*TCombobox*Listbox.font', (FONT, 11))
        for name, bg, fg in [('Start', ACCENT, '#071c19'), ('Stop', '#502936', '#ffdce2'),
                             ('Quiet', FIELD, TEXT)]:
            st.configure(name+'.TButton', background=bg, foreground=fg, borderwidth=0,
                         focusthickness=0, padding=(12,7), font=(FONT,10,'bold'))
            st.map(name+'.TButton', background=[('disabled', '#243145'), ('active', '#47718a' if name=='Quiet' else '#71d9c3' if name=='Start' else '#754254')],
                   foreground=[('disabled', '#8191a7')])
        st.configure('Dark.Vertical.TScrollbar', background=FIELD, troughcolor=BG,
                     bordercolor=BG, arrowcolor=MUTED)

    def label(self, parent, text='', size=10, color=TEXT, bold=False, bg=PANEL, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=(FONT,size,'bold' if bold else 'normal'), **kw)

    def card(self, parent, **kw):
        return tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=EDGE, **kw)

    def build_ui(self):
        shell = tk.Frame(self, bg=BG, padx=24, pady=20)
        shell.pack(fill='both', expand=True)
        header = tk.Frame(shell,bg=BG)
        header.pack(fill='x', pady=(0,18))
        left = tk.Frame(header,bg=BG); left.pack(side='left')
        self.label(left,'FACEVISION  /  BİLGİSAYAR GÖRMESİ',10,ACCENT,True,BG).pack(anchor='w')
        self.label(left,'Gerçek Zamanlı Yüz Tespiti',25,TEXT,True,BG).pack(anchor='w',pady=(4,0))
        self.label(left,'Model karşılaştırma ve performans analizi',11,MUTED,False,BG).pack(anchor='w')
        self.label(header,'VİZE PROJESİ   /   v2.1',10,BLUE,True,BG).pack(side='right',anchor='n')
        self.label(shell,'3 model  •  3 giriş kaynağı  •  CPU / CUDA  •  Görüntü ayrı OpenCV penceresinde açılır.',9,MUTED,False,BG).pack(side='bottom',anchor='w',pady=(10,0))
        body = tk.Frame(shell,bg=BG); body.pack(fill='both',expand=True)
        body.columnconfigure(0,weight=0,minsize=292); body.columnconfigure(1,weight=1)
        body.rowconfigure(0,weight=1)
        sidebar=self.card(body); sidebar.grid(row=0,column=0,sticky='nsew',padx=(0,14))
        side=tk.Frame(sidebar,bg=PANEL,padx=17,pady=12); side.pack(fill='both',expand=True)
        self.label(side,'01  /  Tespit ayarları',14,TEXT,True).pack(anchor='w',pady=(0,12))
        self.model_box=self.combo(side,'MODEL',self.model,list(MODELS))
        self.source_box=self.combo(side,'GİRİŞ KAYNAĞI',self.source,list(SOURCES))
        self.device_box=self.combo(side,'İŞLEM CİHAZI',self.device,['CPU'])
        for box in (self.model_box,self.source_box,self.device_box):
            box.bind('<<ComboboxSelected>>',self.update_options)
        self.info=self.label(side,'',10,MUTED,False,wraplength=248,justify='left',anchor='w')
        self.info.pack(fill='x',pady=(0,5))
        self.file_button=ttk.Button(side,text='Dosya seç…',style='Quiet.TButton',command=self.choose_file)
        self.file_button.pack(fill='x',pady=(0,9))
        self.start_button=ttk.Button(side,text='Tespiti başlat',style='Start.TButton',command=self.start)
        self.start_button.pack(fill='x',pady=(0,8))
        self.stop_button=ttk.Button(side,text='Durdur',style='Stop.TButton',command=self.stop,state='disabled')
        self.stop_button.pack(fill='x',pady=(0,8))
        ttk.Button(side,text='Sonuç klasörünü aç',style='Quiet.TButton',command=self.open_results).pack(fill='x')
        tk.Frame(side,bg=EDGE,height=1).pack(fill='x',pady=9)
        self.label(side,'OTOMATİK PYTHON ORTAMI',9,MUTED,True).pack(anchor='w')
        self.label(side,size=11,color=ACCENT,bold=True,textvariable=self.environment).pack(anchor='w',pady=(5,0))
        self.label(side,'Görüntü penceresinde Q ile çıkış.\nDurdur düğmesi işlemi sonlandırır.',9,MUTED,False,justify='left').pack(anchor='w',side='bottom',pady=(8,0))
        right=tk.Frame(body,bg=BG); right.grid(row=0,column=1,sticky='nsew')
        status=self.card(right); status.pack(fill='x',pady=(0,12))
        sp=tk.Frame(status,bg=PANEL,padx=16,pady=12); sp.pack(fill='x')
        self.status_label=self.label(sp,size=13,color=ACCENT,bold=True,textvariable=self.status)
        self.status_label.pack(anchor='w')
        self.label(sp,size=10,color=MUTED,textvariable=self.session,anchor='w').pack(fill='x',pady=(3,0))
        metrics=tk.Frame(right,bg=BG); metrics.pack(fill='x',pady=(0,12))
        items=[('fps','DETECTION FPS','kare/sn'),('latency','TESPİT SÜRESİ','ms'),('inference','INFERENCE','ms'),('frames','İŞLENEN KARE','adet')]
        for col,(key,title,unit) in enumerate(items):
            metrics.columnconfigure(col,weight=1,uniform='metrics')
            c=self.card(metrics); c.grid(row=0,column=col,sticky='nsew',padx=(0,8) if col<3 else 0)
            p=tk.Frame(c,bg=PANEL,padx=11,pady=12); p.pack(fill='both',expand=True)
            self.label(p,title,9,MUTED,True).pack(anchor='w')
            self.label(p,size=21,color=TEXT,bold=True,textvariable=self.metrics[key]).pack(anchor='w',pady=(4,0))
            self.label(p,unit,9,MUTED).pack(anchor='w')
        terminal=self.card(right); terminal.pack(fill='both',expand=True)
        top=tk.Frame(terminal,bg=PANEL,padx=14,pady=11); top.pack(fill='x')
        self.label(top,'02  /  Çalışma günlüğü',13,TEXT,True).pack(side='left')
        ttk.Button(top,text='Kopyala',style='Quiet.TButton',command=self.copy_log).pack(side='right')
        ttk.Button(top,text='Temizle',style='Quiet.TButton',command=self.clear_log).pack(side='right',padx=(0,7))
        detail=tk.Frame(terminal,bg=PANEL,padx=14,pady=0); detail.pack(fill='x')
        self.label(detail,'Yüz tespiti sayısı:',9,MUTED).pack(side='left')
        self.label(detail,size=10,color=ACCENT,bold=True,textvariable=self.metrics['faces']).pack(side='left',padx=6)
        self.label(detail,'Özet değerler, çıktı geldikçe güncellenir.',9,MUTED).pack(side='right')
        lf=tk.Frame(terminal,bg=PANEL,padx=12,pady=12); lf.pack(fill='both',expand=True)
        self.log=tk.Text(lf,width=40,height=8,highlightthickness=0,wrap='word',font=('Consolas',10),bg='#09111e',fg='#d4e3f4',relief='flat',bd=0,padx=12,pady=10,state='disabled')
        sc=ttk.Scrollbar(lf,orient='vertical',command=self.log.yview,style='Dark.Vertical.TScrollbar')
        sc.pack(side='right',fill='y'); self.log.configure(yscrollcommand=sc.set); self.log.pack(side='left',fill='both',expand=True)
        self.append('FaceVision hazır. Bir model ve giriş kaynağı seçin.\n')

    def combo(self,parent,label,variable,values):
        self.label(parent,label,9,MUTED,True).pack(anchor='w')
        w=ttk.Combobox(parent,textvariable=variable,values=values,state='readonly',style='Modern.TCombobox',width=22,font=(FONT,11))
        w.pack(fill='x',pady=(4,10)); return w

    def update_options(self,event=None):
        choices=device_choices(self.model.get(),self.source.get())
        self.device_box.configure(values=choices)
        if self.device.get() not in choices: self.device.set(choices[0])
        self.environment.set(MODELS[self.model.get()][1])
        source = self.source.get()
        selected = self.selected_files.get(source)
        if source == 'Kamera':
            label = 'Kamera 0 • Canlı görüntü'
        elif selected:
            name = Path(selected).name
            label = 'Seçilen dosya: ' + (name if len(name) <= 50 else name[:47] + '…')
        else:
            default = ROOT / 'data' / ('test.jpg' if source == 'Resim' else 'test_video.mp4')
            label = 'Varsayılan: ' + default.name if default.is_file() else 'Lütfen bir dosya seçin.'
        self.info.configure(text=label)
        self.file_button.configure(state='normal' if source != 'Kamera' and self.process is None else 'disabled')
        if self.process is None and self.session.get()=='Henüz oturum başlatılmadı':
            self.set_status('Hazır',ACCENT)

    def choose_file(self):
        source = self.source.get()
        if source == 'Kamera' or self.process is not None:
            return
        types = ([('Resimler', '*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff')]
                 if source == 'Resim' else [('Videolar', '*.mp4 *.avi *.mov *.mkv *.wmv *.m4v')])
        previous = self.selected_files.get(source)
        folder = Path(previous).parent if previous else ROOT / 'data'
        path = filedialog.askopenfilename(parent=self, title='Resim seç' if source == 'Resim' else 'Video seç',
                                          initialdir=str(folder if folder.is_dir() else ROOT),
                                          filetypes=types + [('Tüm dosyalar', '*.*')])
        if path:
            self.selected_files[source] = str(Path(path).resolve())
            self.update_options()

    def set_status(self,text,color):
        self.status.set(text); self.status_label.configure(fg=color)

    def append(self,text):
        self.log.configure(state='normal'); self.log.insert('end',text); self.log.see('end'); self.log.configure(state='disabled')

    def copy_log(self):
        self.clipboard_clear(); self.clipboard_append(self.log.get('1.0','end-1c'))

    def clear_log(self):
        self.log.configure(state='normal'); self.log.delete('1.0','end'); self.log.configure(state='disabled')

    def start(self):
        if self.process is not None: return
        prefix, environment=MODELS[self.model.get()]; kind=SOURCES[self.source.get()]
        script=SRC/f'{prefix}_{kind}.py'; python=ROOT/environment/'Scripts'/'python.exe'
        if not script.is_file() or not python.is_file():
            messagebox.showerror('Dosya bulunamadı',f'{script}\n{python}'); return
        (ROOT / 'results').mkdir(exist_ok=True)
        selected_file = ''
        if kind != 'camera':
            default = ROOT / 'data' / ('test.jpg' if kind == 'image' else 'test_video.mp4')
            selected_file = self.selected_files.get(self.source.get()) or str(default)
            if not Path(selected_file).is_file():
                messagebox.showerror('Dosya bulunamadı','Lütfen geçerli bir resim veya video seçin.')
                return
        send_choice=prefix=='yolo' and kind=='video'
        choice='1\n' if self.device.get()=='GPU' else '2\n'
        env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'
        self.stop_path=Path(tempfile.gettempdir())/('facevision_stop_'+uuid.uuid4().hex)
        try:
            self.job=DetectionJob()
            self.process=subprocess.Popen([str(python),'-u','-c',WORKER_BOOTSTRAP,self.job.name,str(script),str(self.stop_path),kind,selected_file],cwd=str(SRC),env=env,
                stdin=subprocess.PIPE if send_choice else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except (OSError,AttributeError) as exc:
            self.cleanup(); messagebox.showerror('Başlatma hatası',str(exc)); return
        self.stopping=self.forced=False
        for v in self.metrics.values(): v.set('—')
        self.session.set(f'{self.model.get()}  /  {self.source.get()}  /  {self.device.get()}')
        self.set_status('Çalışıyor…',ACCENT)
        self.start_button.configure(state='disabled'); self.stop_button.configure(state='normal'); self.file_button.configure(state='disabled')
        for box in (self.model_box,self.source_box,self.device_box): box.configure(state='disabled')
        self.append(f'\n--- {self.session.get()} ---\nPython: {python}\n' + (f'Giriş: {selected_file}\n' if selected_file else 'Giriş: Kamera 0\n'))
        process=self.process
        threading.Thread(target=self.read_output,args=(process,),daemon=True).start()
        if send_choice:
            try:
                process.stdin.write(choice); process.stdin.close()
            except OSError as exc: self.append(f'Cihaz seçimi gönderilemedi: {exc}\n')

    def read_output(self,process):
        try:
            with process.stdout:
                for line in process.stdout: self.messages.put(('log',line))
        finally: self.messages.put(('done',process.wait()))

    def cleanup(self):
        if self.job is not None: self.job.close(); self.job=None
        if self.stop_path is not None:
            try: self.stop_path.unlink(missing_ok=True)
            except OSError: pass
            self.stop_path=None

    def poll_messages(self):
        try:
            for _ in range(200):
                kind,value=self.messages.get_nowait()
                if kind=='log':
                    self.append(value)
                    for key,metric in parse_metrics(value).items(): self.metrics[key].set(metric)
                elif kind=='done':
                    self.append(f'\nProcess exit code: {value}\n'); self.process=None; self.cleanup()
                    if self.closing: self.destroy(); return
                    self.set_status('Zorla durduruldu' if self.forced else 'Durduruldu' if self.stopping else 'Tamamlandı' if value==0 else 'Hata oluştu', RED if self.forced or (value!=0 and not self.stopping) else ACCENT)
                    self.stopping=False; self.start_button.configure(state='normal'); self.stop_button.configure(state='disabled')
                    for box in (self.model_box,self.source_box,self.device_box): box.configure(state='readonly')
                    self.update_options()
        except queue.Empty: pass
        self.after(80,self.poll_messages)

    def open_results(self):
        folder=ROOT/'results'; folder.mkdir(exist_ok=True)
        try: os.startfile(str(folder))
        except (OSError,AttributeError) as exc: messagebox.showerror('Klasör açılamadı',str(exc))

    def stop(self):
        if self.process is None or self.stopping: return
        self.stopping=True; self.stop_button.configure(state='disabled'); self.set_status('Durduruluyor…',BLUE)
        self.append('\nDurdurma istendi. Önce normal çıkış deneniyor.\n')
        process=self.process
        try: self.stop_path.touch()
        except OSError: self.force_stop(process); return
        self.after(3500,lambda: self.force_stop(process))

    def force_stop(self,process):
        if self.process is not process or process.poll() is not None: return
        self.forced=True; self.append('Normal çıkış gerçekleşmedi; işlem zorla durduruluyor. Video eksik olabilir.\n')
        if self.job is not None: self.job.close(); self.job=None
        self.after(1000,lambda: self.stop_redirector(process))

    def stop_redirector(self,process):
        if self.process is process and process.poll() is None:
            try: process.kill()
            except OSError: pass

    def close_app(self):
        if self.closing: return
        if self.process is not None:
            if not messagebox.askyesno('Uygulama kapatılsın mı?', 'Aktif tespit durdurulacak. Tamamlanmamış video kaydı eksik olabilir.'): return
            self.closing=True; self.stop(); return
        self.cleanup(); self.destroy()


if __name__=='__main__':
    DetectionApp().mainloop()
