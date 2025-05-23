import json
import os
import shutil
import random
from tqdm import tqdm # Для відображення прогресу

# Конфігурація шляхів
# Шлях до кореневого каталогу розпакованого датасету HRSID
hrs_id_root_dir = "/content/HRSID/" 

# Шлях до каталогу із зображеннями HRSID, враховуючи вкладену папку HRSID_JPG
hrs_id_images_dir = os.path.join(hrs_id_root_dir, "HRSID_JPG", "JPEGImages")

# Вихідний каталог для нового датасету у форматі YOLO
yolo_output_dir = "/content/HRSID_YOLO_Format/"

# Конкретна кількість зображень для кожного набору
train_count = 700
val_count = 150
test_count = 150

#  Функція конвертації COCO JSON в YOLO полігони 
def convert_coco_to_yolo_segmentation(json_path, images_dir, output_labels_dir):
    """
    Конвертує анотації COCO JSON у формат YOLO-полігонів (.txt файли).
    Створює словник відображення image_id -> image_filename.
    Повертає словник {image_filename: [list_of_annotations_for_this_image]}
    та словник {image_filename: (width, height)}.
    """
    print(f"Починаємо конвертацію COCO JSON: {json_path}")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_filename = image_id_to_filename[ann['image_id']]
        if image_filename not in annotations_by_image:
            annotations_by_image[image_filename] = []
        annotations_by_image[image_filename].append(ann)

    # Створюємо вихідний каталог для міток, якщо його немає
    os.makedirs(output_labels_dir, exist_ok=True)

    yolo_label_files = [] # Список для збереження шляхів до створених YOLO-файлів міток

    for image_filename, annotations in tqdm(annotations_by_image.items(), desc="Конвертація анотацій"):
        img_width, img_height = image_id_to_size[annotations[0]['image_id']] # Беремо розмір з першої анотації для зображення
        
        label_file_path = os.path.join(output_labels_dir, os.path.splitext(image_filename)[0] + ".txt")
        
        with open(label_file_path, 'w') as f:
            for ann in annotations:
                # ЗМІНЕНО: Віднімаємо 1 від category_id для 0-індексації YOLO
                category_id = ann['category_id'] - 1 
                
                # Сегментація в COCO може бути у вигляді списку списків (для мультиполігонів)
                segmentation = ann['segmentation']
                
                # Якщо сегментація - це список списків (мультиполігон), об'єднуємо їх
                # YOLOv8 .txt формат очікує один рядок на об'єкт, з одним полігоном.
                # Якщо об'єкт має кілька полігонів, це може бути проблемою.
                # Для простоти візьмемо перший полігон.
                if isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], list):
                    segmentation_coords = segmentation[0] # Беремо перший полігон
                else:
                    segmentation_coords = segmentation

                # ЗМІНЕНО: Додаємо перевірку, чи полігон не порожній
                if not segmentation_coords:
                    continue # Пропускаємо, якщо полігон порожній

                # Нормалізуємо координати полігона
                normalized_coords = []
                for i in range(0, len(segmentation_coords), 2):
                    x = segmentation_coords[i] / img_width
                    y = segmentation_coords[i+1] / img_height
                    normalized_coords.append(f"{x:.6f}")
                    normalized_coords.append(f"{y:.6f}")
                
                # Записуємо у файл: class_id x1 y1 x2 y2 ...
                f.write(f"{category_id} {' '.join(normalized_coords)}\n")
        
        yolo_label_files.append(label_file_path)
    
    print(f"Конвертацію завершено. Створено {len(yolo_label_files)} YOLO-файлів міток.")
    return list(image_id_to_filename.values()) # Повертаємо список всіх імен файлів зображень

#  Функція копіювання файлів до цільового каталогу
def copy_files_to_split_dir(filenames, source_images_dir, dest_images_dir, dest_labels_dir):
    """
    Копіює зображення та їхні відповідні YOLO-мітки до вказаних цільових каталогів.
    Припускає, що файли міток вже були створені в `temp_yolo_labels_dir`.
    """
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    split_name = os.path.basename(os.path.dirname(dest_images_dir))
    print(f"Копіюємо {len(filenames)} файлів до {split_name} набору...")
    for filename in tqdm(filenames, desc=f"Копіювання до {split_name}"):
        # Копіюємо зображення
        src_image_path = os.path.join(source_images_dir, filename)
        dst_image_path = os.path.join(dest_images_dir, filename)
        shutil.copy(src_image_path, dst_image_path)

        # Копіюємо відповідні файли міток (з тимчасового каталогу)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        src_label_path = os.path.join(yolo_output_dir, "temp_all_yolo_labels", label_filename) 
        dst_label_path = os.path.join(dest_labels_dir, label_filename)
        
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            # Це не повинно статися, якщо write_yolo_labels_for_selected_files працює коректно
            print(f"Попередження: Файл міток не знайдено для {filename} за шляхом {src_label_path}. Створюємо порожній.")
            open(dst_label_path, 'a').close()

# Основний блок виконання
if __name__ == "__main__":
    # Визначення шляхів до JSON-файлів анотацій HRSID, враховуючи вкладену папку HRSID_JPG
    hrs_id_train_annotations_json = os.path.join(hrs_id_root_dir, "HRSID_JPG", "annotations", "train2017.json")
    hrs_id_test_annotations_json = os.path.join(hrs_id_root_dir, "HRSID_JPG", "annotations", "test2017.json")

    # --- Зчитуємо та конвертуємо всі анотації в тимчасовий каталог ---
    print("\n--- Зчитуємо та конвертуємо всі анотації в тимчасовий каталог ---")
    temp_all_yolo_labels_dir = os.path.join(yolo_output_dir, "temp_all_yolo_labels")
    os.makedirs(temp_all_yolo_labels_dir, exist_ok=True)

    # Обробка train2017.json
    # convert_coco_to_yolo_segmentation тепер повертає лише список імен файлів
    train_val_image_filenames = convert_coco_to_yolo_segmentation(
        hrs_id_train_annotations_json, hrs_id_images_dir, temp_all_yolo_labels_dir # Записуємо в єдиний тимчасовий каталог
    )
    # Обробка test2017.json
    test_image_filenames = convert_coco_to_yolo_segmentation(
        hrs_id_test_annotations_json, hrs_id_images_dir, temp_all_yolo_labels_dir # Записуємо в єдиний тимчасовий каталог
    )

    # --- Розділяємо та копіюємо файли до фінальних каталогів з потрібною кількістю ---
    print("\n--- Розділяємо та копіюємо файли до фінальних каталогів ---")

    # Перемішуємо тренувальні/валідаційні файли, щоб забезпечити випадковий вибір
    random.shuffle(train_val_image_filenames)
    random.shuffle(test_image_filenames) # Також перемішуємо тестові для випадковості, якщо їх більше 150

    # Вибираємо потрібну кількість файлів
    selected_train_files = train_val_image_filenames[:train_count]
    remaining_train_val_files = train_val_image_filenames[train_count:]
    selected_val_files = remaining_train_val_files[:val_count]
    selected_test_files = test_image_filenames[:test_count]

    # Копіюємо вибрані файли до фінальних каталогів
    copy_files_to_split_dir(selected_train_files, hrs_id_images_dir, 
                            os.path.join(yolo_output_dir, 'train', 'images'),
                            os.path.join(yolo_output_dir, 'train', 'labels'))
    
    copy_files_to_split_dir(selected_val_files, hrs_id_images_dir, 
                            os.path.join(yolo_output_dir, 'val', 'images'),
                            os.path.join(yolo_output_dir, 'val', 'labels'))
    
    copy_files_to_split_dir(selected_test_files, hrs_id_images_dir, 
                            os.path.join(yolo_output_dir, 'test', 'images'),
                            os.path.join(yolo_output_dir, 'test', 'labels'))

    # Очищаємо тимчасовий каталог з усіма мітками
    shutil.rmtree(temp_all_yolo_labels_dir)

    print("\n--- Процес підготовки датасету завершено ---")
    print(f"Ваш датасет у форматі YOLO-полігонів знаходиться за адресою: {yolo_output_dir}")
    print("Тепер ви можете оновити шляхи у вашому основному скрипті навчання YOLOv8:")
    print(f"train_images_dir = \"{os.path.join(yolo_output_dir, 'train', 'images')}\"")
    print(f"train_labels_dir = \"{os.path.join(yolo_output_dir, 'train', 'labels')}\"")
    print(f"val_images_dir = \"{os.path.join(yolo_output_dir, 'val', 'images')}\"")
    print(f"val_labels_dir = \"{os.path.join(yolo_output_dir, 'val', 'labels')}\"")
    print(f"test_images_dir = \"{os.path.join(yolo_output_dir, 'test', 'images')}\"")
    print(f"test_labels_dir = \"{os.path.join(yolo_output_dir, 'test', 'labels')}\"")
    print("Також переконайтеся, що model_type встановлено на 'yolov8s-seg.pt'.")
