# 1.Імпорт необхідних модулів

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import os
import random
from deap import creator, base, tools, algorithms 
import yaml 

# 2.Налаштування для Google Colab

# Монтування Google Диска в Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    colab_env = True
    print("Google Диск успішно змонтовано.")
except ImportError:
    colab_env = False
    print("Не в середовищі Google Colab.")

# 3. Конфігурація

train_images_dir = "/content/HRSID_YOLO_Format/train/images"
train_labels_dir = "/content/HRSID_YOLO_Format/train/labels"
val_images_dir = "/content/HRSID_YOLO_Format/val/images"
val_labels_dir = "/content/HRSID_YOLO_Format/val/labels"
test_images_dir = "/content/HRSID_YOLO_Format/test/images"
test_labels_dir = "/content/HRSID_YOLO_Format/test/labels"

# Вихідний каталог для результатів ГА та фінальної моделі, на Google Диску, якщо в Colab
if colab_env:
    output_dir = '/content/drive/MyDrive/YOLOv8_GA_Results'
else:
    output_dir = 'runs/ga_tuning' # Локальний шлях, якщо не в Colab

# Переконуємося чи вихідний каталог існує
os.makedirs(output_dir, exist_ok=True)

# Шлях для динамічно створеного файлу data.yaml
dynamic_data_yaml_path = os.path.join(output_dir, 'data.yaml')


model_type = 'yolov8s-seg.pt'  # Використовуємо невелику модель сегментації

# Назва класу (HRSID має лише один клас "ship")
class_names = ['ship'] 


# 4. Завантаження даних 

def load_dataset_paths_from_dirs(train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir):
    """
    Використовуємо загалом 1000 фотографій для швидкості обрахувань. 700 - тренувальні, 150 - валідаціні,
    150 - тестові.

    """
    def get_files_from_dir(directory):
        files = []
        if os.path.exists(directory):
            for filename in sorted(os.listdir(directory)): # Сортуємо для забезпечення послідовного порядку
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(directory, filename))
        return files

    # Завантажуємо всі шляхи до зображень з кожного каталогу
    all_train_images = get_files_from_dir(train_img_dir)
    all_val_images = get_files_from_dir(val_img_dir)
    all_test_images = get_files_from_dir(test_img_dir)

    # Обмежуємо кількість зображень для прискорення навчання, якщо датасет великий
    train_images = all_train_images[:700]
    val_images = all_val_images[:150] # 850 - 700 = 150
    test_images = all_test_images[:150] # 1000 - 850 = 150

    # Генеруємо відповідні шляхи до файлів міток
    def get_label_path(image_path, labels_base_dir):
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        return os.path.join(labels_base_dir, name_without_ext + '.txt')

    train_labels = [get_label_path(img_path, train_lbl_dir) for img_path in train_images]
    val_labels = [get_label_path(img_path, val_lbl_dir) for img_path in val_images]
    test_labels = [get_label_path(img_path, test_lbl_dir) for img_path in test_images]

    #Перевіряємо чи все добре

    print(f"Завантажено {len(train_images)} тренувальних зображень.")
    print(f"Завантажено {len(val_images)} валідаційних зображень.")
    print(f"Завантажено {len(test_images)} тестових зображень.")
    print(f"Всього використано зображень: {len(train_images) + len(val_images) + len(test_images)}")


    return {
        'train_images': train_images, 'train_labels': train_labels,
        'val_images': val_images, 'val_labels': val_labels,
        'test_images': test_images, 'test_labels': test_labels,
    }

dataset_info = load_dataset_paths_from_dirs(
    train_images_dir, train_labels_dir,
    val_images_dir, val_labels_dir,
    test_images_dir, test_labels_dir
)
train_images = dataset_info['train_images']
val_images = dataset_info['val_images']
test_images = dataset_info['test_images']
train_labels = dataset_info['train_labels']
val_labels = dataset_info['val_labels']
test_labels = dataset_info['test_labels']


# Створюємо динамічний data.yaml для навчання/валідації YOLOv8
def create_dynamic_data_yaml(output_path, train_img_dir, val_img_dir, test_img_dir, class_names):
    """
    Створює файл data.yaml для навчання YOLOv8.
    """
    data = {
        # Базовий шлях для відносних шляхів у data.yaml тепер має враховувати вкладену папку
        # Якщо ви структуруєте HRSID як /content/HRSID_YOLO_Format/, то path буде /content/HRSID_YOLO_Format/
        'path': os.path.dirname(train_img_dir).rsplit('/', 1)[0], # Отримуємо батьківську директорію для train/images
        'train': os.path.relpath(train_img_dir, os.path.dirname(train_img_dir).rsplit('/', 1)[0]),
        'val': os.path.relpath(val_img_dir, os.path.dirname(train_img_dir).rsplit('/', 1)[0]),
        'test': os.path.relpath(test_img_dir, os.path.dirname(train_img_dir).rsplit('/', 1)[0]),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Динамічний data.yaml створено за адресою: {output_path}")

create_dynamic_data_yaml(
    dynamic_data_yaml_path,
    train_images_dir, val_images_dir, test_images_dir,
    class_names
)

def load_annotations(image_path, image_width, image_height):
    """
    Завантажує анотації (полігони, мітки) для даного зображення з файлу .txt у форматі YOLO-полігонів.
    Очікує, що файли міток вже конвертовані у формат YOLO-полігонів.
    Аргументи:
        image_path (str): Шлях до файлу зображення.
        image_width (int): Ширина зображення.
        image_height (int): Висота зображення.
    Повертає:
        list: Список анотацій. Кожна анотація є кортежем
              (class_id, [x1, y1, x2, y2, ..., xn, yn]) у піксельних координатах.
    """
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    # Припускається, що мітки знаходяться в папці 'labels' паралельно до 'images'
    # наприклад, 'path/to/data/train/images/img.jpg' -> 'path/to/data/train/labels/img.txt'
    label_path = image_path.replace('/images/', '/labels/').replace(base_name, name_without_ext + '.txt')

    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                # Решта частин - це нормалізовані координати полігона
                polygon_coords_norm = parts[1:]
                
                # Перетворюємо нормалізовані координати у піксельні
                polygon_coords_pixel = []
                for i in range(0, len(polygon_coords_norm), 2):
                    x_norm = polygon_coords_norm[i]
                    y_norm = polygon_coords_norm[i+1]
                    polygon_coords_pixel.append(x_norm * image_width)
                    polygon_coords_pixel.append(y_norm * image_height)
                
                annotations.append((class_id, polygon_coords_pixel))
    return annotations

# 5. Налаштування YOLOv8

# Початкова модель для налаштування ГА. Вона буде навчатися кілька разів.
initial_model = YOLO(model_type)

# 6. Функція оцінки
# ----------------------
def evaluate_model_on_set(model_instance, data_yaml_path_for_val, conf_threshold=0.25, iou_threshold=0.45):
    """
    Оцінює модель YOLOv8 на наборі зображень за допомогою вбудованої валідації Ultralytics.
    Це набагато надійніше для розрахунку mAP.
    """
    # Метод val Ultralytics безпосередньо розраховує метрики, такі як mAP
    # Ми повинні переконатися, що модель знає про data.yaml для валідації.
    metrics = model_instance.val(data=data_yaml_path_for_val, conf=conf_threshold, iou=iou_threshold, verbose=False)
    # mAP50-95 є поширеною метрикою. mAP50 також корисний.
    return metrics.box.map, metrics.box.map50 # Повертаємо mAP50-95 та mAP50


# 7. Генетичний алгоритм для налаштування гіперпараметрів

# Визначення типу фітнесу (максимізація) та індивідуального представлення (список чисел з плаваючою комою)
# Перевіряємо, чи класи вже існують, щоб уникнути RuntimeWarning
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# Визначення простору пошуку гіперпараметрів з науково обґрунтованих діапазонів
# Це загальні діапазони для гіперпараметрів навчання YOLOv8.
HYPERPARAMETER_RANGES = {
    'epochs': (5, 20), # Кількість епох для внутрішнього навчання ГА 
    'batch': (4, 16), # Розмір пакету 
    'imgsz': (320, 640), # Розмір зображення (має ділитися на 32)
    'lr0': (0.001, 0.05), # Початкова швидкість навчання
    'lrf': (0.0001, 0.01), # Кінцева швидкість навчання (коефіцієнт lr0)
    'momentum': (0.9, 0.98), # Момент SGD
    'weight_decay': (0.0001, 0.001), # Зменшення ваги
    'hsv_h': (0.0, 0.1), # Аугментація відтінку HSV
    'hsv_s': (0.0, 0.9), # Аугментація насиченості HSV
    'hsv_v': (0.0, 0.9), # Аугментація значення HSV
    'degrees': (0.0, 10.0), # Градуси випадкового повороту
    'translate': (0.0, 0.2), # Випадковий зсув
    'scale': (0.5, 1.0), # ВИПРАВЛЕНО: Випадковий масштаб, діапазон 0.0-1.0 для Ultralytics
    'shear': (0.0, 10.0), # Випадковий зсув
    'perspective': (0.0, 0.001), # Перспективне перетворення
    'flipud': (0.0, 0.5), # Перевернути вгору-вниз
    'fliplr': (0.0, 0.5), # Перевернути вліво-вправо
    'mosaic': (0.0, 1.0), # Аугментація мозаїки
    'mixup': (0.0, 0.1), # Аугментація Mixup
    'copy_paste': (0.0, 0.1), # Аугментація Copy-paste
}

# Перетворення словника діапазонів на список для індексування DEAP
HYPERPARAMETER_KEYS = list(HYPERPARAMETER_RANGES.keys())
HYPERPARAMETER_BOUNDS = [HYPERPARAMETER_RANGES[key] for key in HYPERPARAMETER_KEYS]

# Параметри ГА
POPULATION_SIZE = 5
NUM_GENERATIONS = 3 # Мале число, щоб пришвидшити обрахунки
CXPB = 0.7  # Ймовірність кросоверу
MUTPB = 0.2 # Ймовірність мутації

def generate_individual_params():
    """Генерує випадковий набір гіперпараметрів (як словник)."""
    params = {}
    for param_name, (min_val, max_val) in HYPERPARAMETER_RANGES.items():
        if param_name in ['epochs', 'batch', 'imgsz']: # Цілочисельні параметри
            params[param_name] = random.randint(int(min_val), int(max_val))
            if param_name == 'imgsz': # Переконайтеся, що imgsz ділиться на 32
                params[param_name] = (params[param_name] // 32) * 32
        else: # Параметри з плаваючою комою
            params[param_name] = random.uniform(min_val, max_val)
    return params

def evaluate_fitness_deap(individual_list, model_type, data_yaml_path):
    """
    Оцінює пристосованість індивідуума (набору гіперпараметрів).
    Це передбачає навчання нової моделі YOLOv8s та оцінку її mAP на валідаційному наборі.
    """
    # Перетворити список індивідуума DEAP на словник гіперпараметрів
    individual_params = {
        HYPERPARAMETER_KEYS[i]: individual_list[i]
        for i in range(len(HYPERPARAMETER_KEYS))
    }
    
    # Забезпечити, що цілочисельні параметри є цілими числами
    for p_name in ['epochs', 'batch', 'imgsz']:
        if p_name in individual_params:
            individual_params[p_name] = int(individual_params[p_name])
            if p_name == 'imgsz':
                individual_params[p_name] = (individual_params[p_name] // 32) * 32


    print(f"\n--- Оцінка індивідуума: {individual_params} ---")
    
    temp_model = YOLO(model_type)

    try:
        temp_model.train(
            data=data_yaml_path,
            epochs=individual_params['epochs'],
            imgsz=individual_params['imgsz'],
            batch=individual_params['batch'],
            lr0=individual_params['lr0'],
            lrf=individual_params['lrf'],
            momentum=individual_params['momentum'],
            weight_decay=individual_params['weight_decay'],
            hsv_h=individual_params['hsv_h'],
            hsv_s=individual_params['hsv_s'],
            hsv_v=individual_params['hsv_v'],
            degrees=individual_params['degrees'],
            translate=individual_params['translate'],
            scale=individual_params['scale'],
            shear=individual_params['shear'],
            perspective=individual_params['perspective'],
            flipud=individual_params['flipud'],
            fliplr=individual_params['fliplr'],
            mosaic=individual_params['mosaic'],
            mixup=individual_params['mixup'],
            copy_paste=individual_params['copy_paste'],
            project=output_dir, # Змінено на output_dir
            name=f"ga_run_{random.randint(1000, 9999)}", # Унікальне ім'я запуску
            val=True, # Виконувати валідацію під час навчання
            patience=5, # Терпіння для раннього завершення для запусків ГА
            verbose=False # Зробити вивід навчання стислим
        )
        val_metrics = temp_model.val(data=data_yaml_path, conf=0.25, iou=0.45, verbose=False)
        mAP50_95 = val_metrics.box.map
        mAP50 = val_metrics.box.map50
        print(f"Валідація mAP50-95: {mAP50_95:.4f}, mAP50: {mAP50:.4f}")
        return mAP50_95, # Повертаємо mAP50-95 як кортеж для DEAP
    except Exception as e:
        print(f"Помилка під час навчання/оцінки для індивідуума: {e}")
        return 0.0, # Повертаємо 0 пристосованості для невдалих запусків

def tune_hyperparameters_with_ga_deap(model_type, data_yaml_path):
    """
    Налаштовує гіперпараметри моделі YOLOv8 за допомогою генетичного алгоритму DEAP.
    """
    toolbox = base.Toolbox()

    # Атрибути: генеруємо випадкові значення в межах діапазонів
    for i, (key, (min_val, max_val)) in enumerate(HYPERPARAMETER_RANGES.items()):
        if key in ['epochs', 'batch', 'imgsz']:
            toolbox.register(f"attr_hyperparam_{i}", random.randint, int(min_val), int(max_val))
        else:
            toolbox.register(f"attr_hyperparam_{i}", random.uniform, min_val, max_val)

    # Комбінуємо атрибути для створення індивідуума
    individual_attrs = [getattr(toolbox, f"attr_hyperparam_{i}") for i in range(len(HYPERPARAMETER_KEYS))]
    toolbox.register("individual", tools.initCycle, creator.Individual, individual_attrs, n=1)

    # Реєструємо функції для популяції, оцінки, кросоверу та мутації
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness_deap, model_type=model_type, data_yaml_path=data_yaml_path)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Змішаний кросовер
    # Мутація Гаусса: mu=середнє, sigma=стандартне відхилення, indpb=ймовірність мутації гена
    # sigma для кожного параметра має бути налаштована відповідно до його діапазону
    sigmas = [(bounds[1] - bounds[0]) * 0.1 for bounds in HYPERPARAMETER_BOUNDS] # 10% від діапазону
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigmas, indpb=0.1) # Ймовірність мутації гена 0.1
    toolbox.register("select", tools.selTournament, tournsize=3) # Турнірний відбір

    # Ініціалізація популяції
    population = toolbox.population(n=POPULATION_SIZE)

    # Запускаємо генетичний алгоритм
    # `hof` (Hall of Fame) зберігає найкращих індивідуумів
    hof = tools.HallOfFame(1) # ВИПРАВЛЕНО: HallOfFame з великої 'O'
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, log = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB,
                                          ngen=NUM_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # Найкращий індивідуум знаходиться в hof[0]
    best_individual_list = hof[0]
    best_individual_params = {
        HYPERPARAMETER_KEYS[i]: best_individual_list[i]
        for i in range(len(HYPERPARAMETER_KEYS))
    }
    # Забезпечити, що цілочисельні параметри є цілими числами
    for p_name in ['epochs', 'batch', 'imgsz']:
        if p_name in best_individual_params:
            best_individual_params[p_name] = int(best_individual_params[p_name])
            if p_name == 'imgsz':
                best_individual_params[p_name] = (best_individual_params[p_name] // 32) * 32

    print(f"\n--- Налаштування ГА завершено ---")
    print(f"Знайдено найкращі гіперпараметри: {best_individual_params}")
    print(f"Найкраща пристосованість валідації (mAP50-95): {hof[0].fitness.values[0]:.4f}")
    
    return best_individual_params


# 8. Візуалізація проблемних виявлень/сегментацій

def visualize_problematic_images(model_instance, image_paths, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Візуалізує зображення з хибними позитивними, хибними негативними або поганими сегментаціями.
    Зберігає ці зображення у вказаному вихідному каталозі.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    box_annotator = sv.BoundingBoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_size=0.7, text_thickness=1, text_color=sv.Color.white())
    mask_annotator = sv.MaskAnnotator(opacity=0.5) 

    problem_count = 0
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Попередження: Не вдалося прочитати зображення {image_path}. Пропускаємо.")
            continue

        img_height, img_width, _ = image.shape
        
        # Отримуємо ground truth виявлення
        gt_annotations = load_annotations(image_path, img_width, img_height)
        gt_xy = []
        gt_class_ids = []
        for class_id, polygon_coords_pixel in gt_annotations:
            # Для Supervision.Detections з полігонами потрібні xyxy для bounding box
            # та mask для полігона. Обчислюємо xyxy з полігона.
            polygon_np = np.array(polygon_coords_pixel).reshape(-1, 2)
            x_min, y_min = np.min(polygon_np, axis=0)
            x_max, y_max = np.max(polygon_np, axis=0)
            gt_xy.append([x_min, y_min, x_max, y_max])
            gt_class_ids.append(class_id)
        
        gt_detections = sv.Detections(
            xyxy=np.array(gt_xy),
            class_id=np.array(gt_class_ids)
            # Примітка: Маски GT не додаються до gt_detections тут, але можуть бути використані окремо
        )

        # Отримуємо прогнози моделі
        results = model_instance.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        if results[0].boxes is None:
            pred_detections = sv.Detections.empty()
        else:
            # Для моделі сегментації, results[0].masks буде містити маски
            pred_detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                mask=results[0].masks.xyxy.cpu().numpy() if results[0].masks is not None else None
            )
        
        # Зіставляємо прогнози з ground truth (для виявлення)
        if len(gt_detections) > 0 and len(pred_detections) > 0:
            matches = sv.match_detections(gt_detections, pred_detections, iou_threshold=iou_threshold)
            
            # Виявляємо хибні позитивні (прогнози без відповідності)
            false_positives_indices = np.where(matches.iou == -1)[0]
            false_positives = pred_detections[false_positives_indices]

            # Виявляємо хибні негативні (ground truths без відповідності)
            false_negatives_indices = np.where(matches.match_iou == -1)[0]
            false_negatives = gt_detections[false_negatives_indices]

            # Виявляємо погані відповідності IoU (правильно виявлено, але погане перекриття)
            poor_iou_matches_indices = np.where((matches.iou != -1) & (matches.iou < 0.7))[0]
            poor_iou_preds = pred_detections[poor_iou_matches_indices]
            poor_iou_gts = gt_detections[matches.match_idx[poor_iou_matches_indices]]

            is_problematic = (len(false_positives) > 0 or
                              len(false_negatives) > 0 or
                              len(poor_iou_matches_indices) > 0)
        else:
            is_problematic = (len(gt_detections) > 0 and len(pred_detections) == 0) or \
                             (len(gt_detections) == 0 and len(pred_detections) > 0)
            false_positives = pred_detections if len(gt_detections) == 0 else sv.Detections.empty()
            false_negatives = gt_detections if len(pred_detections) == 0 else sv.Detections.empty()
            poor_iou_preds = sv.Detections.empty()
            poor_iou_gts = sv.Detections.empty()


        if is_problematic:
            problem_count += 1
            annotated_image = image.copy()

            # Анотуємо прогнозовані маски (для сегментації)
            if pred_detections.mask is not None:
                annotated_image = mask_annotator.annotate(annotated_image, detections=pred_detections)
            
            # Анотуємо хибні позитивні (наприклад, зелені рамки)
            if len(false_positives) > 0:
                labels_fp = [f"ХП: {class_names[cid]}" for cid in false_positives.class_id]
                annotated_image = box_annotator.annotate(annotated_image, detections=false_positives, color=sv.Color.green())
                annotated_image = label_annotator.annotate(annotated_image, detections=false_positives, labels=labels_fp, color=sv.Color.green())

            # Анотуємо хибні негативні (наприклад, червоні рамки для відсутнього GT)
            if len(false_negatives) > 0:
                labels_fn = [f"ХН: {class_names[cid]}" for cid in false_negatives.class_id]
                annotated_image = box_annotator.annotate(annotated_image, detections=false_negatives, color=sv.Color.red())
                annotated_image = label_annotator.annotate(annotated_image, detections=false_negatives, labels=labels_fn, color=sv.Color.red())

            # Анотуємо погані відповідності IoU (наприклад, помаранчеві рамки для прогнозу, сині для GT)
            if len(poor_iou_preds) > 0:
                labels_poor_iou_pred = [f"Пог. IoU Прогн: {class_names[cid]}" for cid in poor_iou_preds.class_id]
                annotated_image = box_annotator.annotate(annotated_image, detections=poor_iou_preds, color=sv.Color.orange())
                annotated_image = label_annotator.annotate(annotated_image, detections=poor_iou_preds, labels=labels_poor_iou_pred, color=sv.Color.orange())

                labels_poor_iou_gt = [f"Пог. IoU GT: {class_names[cid]}" for cid in poor_iou_gts.class_id]
                annotated_image = box_annotator.annotate(annotated_image, detections=poor_iou_gts, color=sv.Color.blue())
                annotated_image = label_annotator.annotate(annotated_image, detections=poor_iou_gts, labels=labels_poor_iou_gt, color=sv.Color.blue())


            # Зберігаємо проблемне зображення
            output_path = os.path.join(output_dir, f"problematic_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, annotated_image)
            print(f"Збережено проблемне зображення: {output_path}")

    print(f"\nЗавершено візуалізацію проблемних зображень. Всього проблемних зображень: {problem_count}")


# 9. Основний блок

if __name__ == "__main__":
    print("Запускаємо налаштування гіперпараметрів YOLOv8s за допомогою генетичного алгоритму...")
    print(f"Використовуємо модель: {model_type}")

    # Крок 1: Налаштовуємо гіперпараметри за допомогою генетичного алгоритму на валідаційному наборі.
    # Це внутрішньо навчатиме кілька невеликих моделей.
    best_hyperparameters = tune_hyperparameters_with_ga_deap(model_type, dynamic_data_yaml_path)
    print("\n--- Результати налаштування ГА ---")
    print("Знайдено найкращі гіперпараметри:", best_hyperparameters)

    # Крок 2: Навчаємо фінальну модель YOLOv8s з найкращими гіперпараметрами
    print("\n--- Навчання фінальної моделі з найкращими гіперпараметрами ---")
    final_model = YOLO(model_type)

    # Метод train Ultralytics автоматично зберігає найкращу модель
    final_train_results = final_model.train(
        data=dynamic_data_yaml_path,
        epochs=best_hyperparameters['epochs'],
        imgsz=best_hyperparameters['imgsz'],
        batch=best_hyperparameters['batch'],
        lr0=best_hyperparameters['lr0'],
        lrf=best_hyperparameters['lrf'],
        momentum=best_hyperparameters['momentum'],
        weight_decay=best_hyperparameters['weight_decay'],
        hsv_h=best_hyperparameters['hsv_h'],
        hsv_s=best_hyperparameters['hsv_s'],
        hsv_v=best_hyperparameters['hsv_v'],
        degrees=best_hyperparameters['degrees'],
        translate=best_hyperparameters['translate'],
        scale=best_hyperparameters['scale'],
        shear=best_hyperparameters['shear'],
        perspective=best_hyperparameters['perspective'],
        flipud=best_hyperparameters['flipud'],
        fliplr=best_hyperparameters['fliplr'],
        mosaic=best_hyperparameters['mosaic'],
        mixup=best_hyperparameters['mixup'],
        copy_paste=best_hyperparameters['copy_paste'],
        project=output_dir,
        name='final_tuned_run', # Ім'я для фінального запуску навчання
        val=True, # Виконувати валідацію під час фінального навчання
        # save=True, # Зберігати контрольні точки (за замовчуванням True)
        # save_period=10 # Зберігати кожні 10 епох
    )

    # Крок 3: Оцінюємо фінальну модель на тестовому наборі
    print("\n--- Оцінка фінальної моделі на тестовому наборі ---")
    # Метод val використовуватиме тестовий поділ, визначений у DYNAMIC_DATA_YAML_PATH
    test_map50_95, test_map50 = evaluate_model_on_set(final_model, dynamic_data_yaml_path)
    print(f"Фінальна модель. Тестовий набір mAP50-95: {test_map50_95:.4f}")
    print(f"Фінальна модель. Тестовий набір mAP50: {test_map50:.4f}")

    # Крок 4: Візуалізуємо проблемні зображення з тестового набору
    print("\n--- Візуалізація проблемних зображень з тестового набору ---")
    # Використовуємо список test_images, а не data_yaml_path для візуалізації
    visualize_problematic_images(final_model, test_images, os.path.join(output_dir, 'problematic_test_images'))

    print("\nПроцес завершено. Перевірте папку 'YOLOv8_GA_Results' на вашому Google Диску на наявність результатів та проблемних зображень.")

