"""تنظیمات مرکزی پایپ‌لاین طبقه‌بندی حیوانات خانگی (Pet).

تمام مقادیر قابل تنظیم (مسیرها، seed، ابعاد ورودی، batch، epoch، lr و ...)
در همین فایل یک‌جا تعریف می‌شوند تا سایر ماژول‌ها فقط از همین تنظیمات استفاده
کنند و رفتار کل پایپ‌لاین قابل بازتولید باشد.
"""
import os  # برای خواندن متغیر محیطی PET_DATA_DIR
import pathlib  # کار با مسیر فایل به‌صورت مستقل از سیستم‌عامل

# ------------------------------------------------------------------
# بخش Reproducibility (قابلیت بازتولید نتایج)
# ------------------------------------------------------------------
SEED = 42  # عدد seed که در random/numpy/torch اعمال می‌شود
# الگوریتم‌های deterministic روی برخی هسته‌های CPU غیرقطعی هستند (مثلاً conv
# backward)، بنابراین فلگ deterministic را خاموش نگه می‌داریم ولی seedها را
# تنظیم می‌کنیم و cudnn.benchmark را غیرفعال می‌کنیم. توضیحات کامل در گزارش.
USE_DETERMINISTIC = False  # فعال نکردن حالت کامل قطعی برای جلوگیری از خطای CPU

# ------------------------------------------------------------------
# بخش Paths (مسیرها)
# ------------------------------------------------------------------
# مسیر ریشه repository (یک پوشه بالاتر از این فایل)
ROOT = pathlib.Path(__file__).resolve().parents[1]
# مسیر پوشه دیتاست Oxford-IIIT Pet. اگر متغیر محیطی PET_DATA_DIR تنظیم
# شده باشد از آن استفاده می‌کنیم؛ در غیر این صورت مسیر پیش‌فرض repo/data
DATA_DIR = pathlib.Path(os.environ.get("PET_DATA_DIR", ROOT / "data" / "oxford-iiit-pet"))
# کلاس OxfordIIITPet در torchvision انتظار دارد root مسیر پدرِ پوشه
# oxford-iiit-pet باشد، نه خود آن پوشه؛ بنابراین این متغیر را هم نگه
# می‌داریم تا در make_splits و data.py استفاده شود.
DATA_ROOT_PARENT = DATA_DIR.parent
# پوشه ذخیره فایل‌های تقسیم‌بندی train/val/test
SPLITS_DIR = ROOT / "splits"
# پوشه ذخیره نتایج (logs، eval، figures)
RESULTS_DIR = ROOT / "results"
# پوشه ذخیره checkpointها (وزن‌های آموزش‌دیده)
CHECKPOINT_DIR = ROOT / "checkpoints"
# پوشه ذخیره تصاویر شکل‌ها (confusion matrix، نمونه‌ها)
FIGURES_DIR = RESULTS_DIR / "figures"

# این حلقه مطمئن می‌شود که پوشه‌های results و checkpoints و splits و figures
# واقعاً وجود دارند (اگر نبودند ساخته می‌شوند)
for d in (SPLITS_DIR, RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)  # ساختن پوشه والد و خود پوشه

# ------------------------------------------------------------------
# بخش Data (تنظیمات داده)
# ------------------------------------------------------------------
NUM_CLASSES = 37  # تعداد کلاس‌ها (نژاد گربه/سگ): 37 نژاد
IMAGE_SIZE = 224  # ابعاد ورودی MobileNetV3-Small (224×224)
BATCH_SIZE = 64  # اندازه batch برای آموزش و ارزیابی
VAL_FRACTION = 0.15  # کسر تقسیم‌بندی stratified برای validation (15٪ trainval)
# در این میزبان /dev/shm فقط 64 مگابایت است؛ برای اشتراک tensor بین worker
# processها خیلی کوچک است. با num_workers=0 داده‌بارگذاری در پردازش اصلی انجام
# می‌شود؛ کندتر ولی مطمئن است.
NUM_WORKERS = 0  # تعداد workerهای DataLoader (0 یعنی همان پردازش اصلی)

# ------------------------------------------------------------------
# بخش Training (هایپرپارامترهای آموزش)
# ------------------------------------------------------------------
EPOCHS_FULL = 8  # تعداد epoch برای رویکرد 1 (full fine-tuning)
EPOCHS_HEAD = 15  # تعداد epoch برای رویکرد 2 (آموزش فقط head) — با early stopping
LR_FULL = 1e-4  # نرخ یادگیری برای رویکرد 1 (کوچک تا وزن‌های ImageNet حفظ شوند)
LR_HEAD = 1e-3  # نرخ یادگیری برای رویکرد 2 (بزرگ‌تر چون فقط head آموزش می‌بیند)
WEIGHT_DECAY = 1e-4  # ضریب L2 regularization در بهینه‌ساز AdamW
PATIENCE = 4  # تعداد epoch بدون بهبود در val_acc قبل از early stopping

# رویکرد 3 (استخراج ویژگی + scikit-learn) — نوع طبقه‌بند پیش‌فرض
FEATURE_CLASSIFIER = "logreg"  # انتخاب بین logreg / svm_linear / random_forest
