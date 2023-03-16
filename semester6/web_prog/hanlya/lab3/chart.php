<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

$query = 'select reader.name, count(*) from issued_books as book inner join reader on book.reader = reader.id group by reader.name;';

$result = pg_query($db, $query) or die('Ошибка запроса: ' . pg_last_error());

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    $res_arr[$line['name']] = $line['count'];
}

// echo var_dump($res_arr);

$rows = array();

$readers = array_keys($res_arr);

foreach ($res_arr as $value) {
    array_push($rows, $value);
}

$rowCount = count($readers);

// Ширина интервала между столбцами
$rowInterval = 50;

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

$candleColor = imagecolorallocate($img, 134, 115, 161);

for ($i = 0, $y1 = $imageHeight, $x1 = $rowInterval / 2; $i < count($rows); $i++) {
    // Формируем случайный цвет для каждого из столбца
    // Нормирование высоты столбца
    $y2 = $y1 + $topPadding - $rows[$i] * $imageHeight / $rowMaxHeight;
    // Определение второй координаты столбца
    $x2 = $x1 + $rowWidth;
    // Отрисовываем столбец
    imagefilledrectangle($img, $x1, $y1, $x2, $y2, $candleColor);
    imagestring($img, 5, $x2 - ($rowWidth / 2), $y2 - 20, $rows[$i], $textColor);

    imagettftext($img, 14, 0, $x1, $height - ($height - $imageHeight) + 20, 5, realpath('.') . "/roboto.ttf", $readers[$i]);

    $x1 = $x2 + $rowInterval;
}

// Выводим изображение в браузер, в формате png
header("Content-type: image/png");
imagepng($img);
?>