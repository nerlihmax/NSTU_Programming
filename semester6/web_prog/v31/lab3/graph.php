<?php
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v31');

$query = 'SELECT 
            operation.name, 
            count(*) 
        from technological_map 
        inner join operation on technological_map.operation = operation.id 
        group by operation.name;';

$result = pg_query($db, $query) or die('Ошибка запроса: ' . pg_last_error());

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    $res_arr[$line['name']] = $line['count'];
}

$rows = array();
$readers = array_keys($res_arr);
foreach ($res_arr as $value) {
    array_push($rows, $value);
}

$rowCount = count($readers);
$rowInterval = 50;
$rowWidth = 150;

$width = ($rowWidth + $rowInterval) * $rowCount;
$height = 350;
$imageHeight = 300;
$rowMaxHeight = max($rows);
$topPadding = 30;

$img = imagecreatetruecolor($width, $height);
$white = imagecolorallocate($img, 255, 255, 200);
$textColor = imagecolorallocate($img, 0, 0, 0);
$candleLegendColor = imagecolorallocate($img, 255, 255, 255);
imagefill($img, 0, 0, $white);
$candleColor = imagecolorallocate($img, 200, 10, 10);
for ($i = 0, $y1 = $imageHeight, $x1 = $rowInterval / 2; $i < count($rows); $i++) {
    $y2 = $y1 + $topPadding - $rows[$i] * $imageHeight / $rowMaxHeight;
    $x2 = $x1 + $rowWidth;
    imagefilledrectangle($img, $x1, $y1, $x2, $y2, $candleColor);
    imagestring($img, 5, $x2 - ($rowWidth / 2), $y2 - 20, $rows[$i], $textColor);
    imagettftext($img, 14, 90, ($x1 + $rowWidth / 2), $height - ($height - $imageHeight) - 10, $candleLegendColor, realpath('.') . "/montserrat.ttf", $readers[$i]);
    $x1 = $x2 + $rowInterval;
}
header("Content-type: image/png");
imagepng($img);
?>