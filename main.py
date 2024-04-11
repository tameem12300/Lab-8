import cv2
import numpy as np

def task_1():
    # Загрузить изображение
    img = cv2.imread('images/img4.jpg')
    # Убедиться, что изображение загружено
    if img is None:
        print("Изображение не найдено или путь неверен.")
        return
    # Изменить размер изображения до 500x500
    img = cv2.resize(img, (500, 500))
    # Извлечь синий канал
    b_channel = img[:,:,0]
    # Создать новое изображение только с синим каналом
    b_img = np.zeros(img.shape, dtype=np.uint8)
    b_img[:,:,0] = b_channel
    # Показать результат
    cv2.imshow('BLUE CHANNEL', b_img)
    # Ожидание нажатия клавиши
    cv2.waitKey(0)
    # Закрыть все окна
    cv2.destroyAllWindows()

def task_2():
    # Начать захват видео
    cap = cv2.VideoCapture(0)
    # Проверить успешность открытия камеры
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        return
    # Получить координату середины кадра по оси X
    middle_x = cap.get(3) / 2

    while True:
        # Считать новый кадр
        ret, frame = cap.read()
        if not ret:
            print("Не удалось захватить кадр")
            break

        # Преобразовать кадр в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Применить инверсию двоичного порога
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # Найти контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Найти наибольший контур
            c = max(contours, key=cv2.contourArea)
            # Получить прямоугольник, ограничивающий наибольший контур
            x, _, w, _ = cv2.boundingRect(c)
            # Проверить, находится ли центр контура в правой половине кадра
            if x + w/2 > middle_x:
                cv2.putText(frame, 'MARKER TO THE RIGHT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показать кадр
        cv2.imshow('FRAME', frame)

        # Прервать цикл по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освободить захват видео и закрыть окна
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Убрать комментарий с функции, которую хотите выполнить
    # task_1()
    task_2()