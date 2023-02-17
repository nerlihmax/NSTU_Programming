<?php

require_once 'config.php';

ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

$dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db)
    or die('Не удалось соединиться: ' . pg_last_error());

$query = 'select city.name, count(*) from ads inner join cities as city on ads.city = city.id group by city.name;';

$result = pg_query($dbconn, $query) or die('Ошибка запроса: ' . pg_last_error());

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    $res_arr[$line['name']] = $line['count'];
}

// echo var_dump($res_arr);

// Значение столбцов от 0 до 100

$rows = array();

$cities = array_keys($res_arr);

foreach ($res_arr as $value) {
    array_push($rows, $value);
}

// echo var_dump($rows);
// echo var_dump($cities);

$rowCount = count($cities);

// Ширина интервала между столбцами
$rowInterval = 20;

// Ширина одного столбца
$rowWidth = 150;

// Ширина изображения
$width = ($rowWidth + $rowInterval) * $rowCount;

// Высота изображения
$height = 350;

$imageHeight = 300;

$rowMaxHeight = max($rows);

$topPadding = 30;

// Создаем пустое изображение
$img = imagecreatetruecolor($width, $height);

// Заливаем изображение белым цветом
$white = imagecolorallocate($img, 255, 255, 255);

$textColor = imagecolorallocate($img, 255, 0, 0);

imagefill($img, 0, 0, $white);

$candleColor = imagecolorallocate($img, 255, 0, 255);

for ($i = 0, $y1 = $imageHeight, $x1 = $rowInterval / 2; $i < count($rows); $i++) {
    // Формируем случайный цвет для каждого из столбца
    // Нормирование высоты столбца
    $y2 = $y1 + $topPadding - $rows[$i] * $imageHeight / $rowMaxHeight;
    // Определение второй координаты столбца
    $x2 = $x1 + $rowWidth;
    // Отрисовываем столбец
    imagefilledrectangle($img, $x1, $y1, $x2, $y2, $candleColor);
    imagestring($img, 5, $x2 - ($rowWidth / 2), $y2 - 20, $rows[$i], $textColor);

    imagettftext($img, 14, 0, $x1, $height - ($height - $imageHeight) + 20, 5, realpath('.') . "/roboto.ttf", $cities[$i]);

    // imagestring($img, 5, $x2 - ($rowWidth / 2), $height - ($height - $imageHeight) + 20, $cityName, $textColor);
    // Между столбцами создаем интервал в $row_interval пикселей
    $x1 = $x2 + $rowInterval;
}
// Выводим изображение в браузер, в формате GIF
header("Content-type: image/png");
imagepng($img);
?>