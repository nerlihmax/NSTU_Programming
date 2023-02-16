<?php

require_once 'config.php';

$dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db)
    or die('Не удалось соединиться: ' . pg_last_error());

$query = 'select roominess, count(*) from ads group by roominess;';

$result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    $res_arr[$line['roominess']] = $line['count'];
}

// Значение столбцов от 0 до 100
$rows = array($res_arr[1] | 0, $res_arr[2] | 0, $res_arr[3] | 0, $res_arr[4] | 0);


// Ширина изображения
$width = 600;
// Высота изображения
$height = 350;

$imageHeight = 300;

// Ширина одного столбца
$rowWidth = 80;
$rowMaxHeight = max($rows);
$rowCount = 4;

$topPadding = 30;

// Ширина интервала между столбцами
$rowInterval = (($width / $rowCount) - $rowWidth);

// Создаем пустое изображение
$img = imagecreatetruecolor($width, $height);

// Заливаем изображение белым цветом
$white = imagecolorallocate($img, 255, 255, 255);
$textColor = imagecolorallocate($img, 255, 255, 255);
imagefill($img, 0, 0, $white);


for ($i = 0, $y1 = $imageHeight, $x1 = $rowInterval / 2; $i < count($rows); $i++) {
    // Формируем случайный цвет для каждого из столбца
    $color = imagecolorallocate($img, rand(0, 255), rand(0, 255), rand(0, 255));
    // Нормирование высоты столбца
    $y2 = $y1 + $topPadding - $rows[$i] * $imageHeight / $rowMaxHeight;
    // Определение второй координаты столбца
    $x2 = $x1 + $rowWidth;
    // Отрисовываем столбец
    imagefilledrectangle($img, $x1, $y1, $x2, $y2, $color);
    imagestring($img, 5, $x2 - ($rowWidth / 2), $height - ($height - $imageHeight) + 20, $i + 1, $textcolor);
    imagestring($img, 5, $x2 - ($rowWidth / 2), $y2 - 20, $rows[$i], $textcolor);
    // Между столбцами создаем интервал в $row_interval пикселей
    $x1 = $x2 + $rowInterval;
}

// Выводим изображение в браузер, в формате GIF
header("Content-type: image/png");
imagepng($img);
?>