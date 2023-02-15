<?php

require_once 'config.php';

$dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db)
    or die('Не удалось соединиться: ' . pg_last_error());

$query = 'select roominess, count(*) from ads group by roominess;';

$result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

$result = pg_fetch_array($result, null, PGSQL_ASSOC);

// Значение столбцов от 0 до 100
$rows = array(80, 75, 53, 32, 20);

// Ширина изображения
$width = 600;
// Высота изображения
$height = 300;
// Ширина одного столбца
$rowWidth = 80;

$rowCount = 4;

// Ширина интервала между столбцами
$rowInterval = (($width / $rowCount) - $rowWidth);

// Создаем пустое изображение
$img = imagecreatetruecolor($width, $height);

// Заливаем изображение белым цветом
$white = imagecolorallocate($img, 255, 255, 255);
imagefill($img, 0, 0, $white);

for ($i = 0, $y1 = $height, $x1 = $rowInterval / 2; $i < count($rows); $i++) {
    // Формируем случайный цвет для каждого из столбца
    $color = imagecolorallocate($img, rand(0, 255), rand(0, 255), rand(0, 255));
    // Нормирование высоты столбца
    $y2 = $y1 - $rows[$i] * $height / 100;
    // Определение второй координаты столбца
    $x2 = $x1 + $rowWidth;
    // Отрисовываем столбец
    imagefilledrectangle($img, $x1, $y1, $x2, $y2, $color);
    // Между столбцами создаем интервал в $row_interval пикселей
    $x1 = $x2 + $rowInterval;
}

// Выводим изображение в браузер, в формате GIF
header("Content-type: image/png");
imagepng($img);
?>