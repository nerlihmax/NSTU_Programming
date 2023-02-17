<?php

require_once 'config.php';

ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

$dbconn = pg_connect('host=' . $host . ' port=' . $port . ' user=' . $user . ' password=' . $password . ' dbname=' . $db) or die('Не удалось соединиться: ' . pg_last_error());

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$city = $queries["city"] or die("No id provided");

$query = 'select ad_types.type, cities.name as city, ads.address, ads.roominess, ads.price, ads.created_at from ads inner join cities on cities.id = ads.city inner join ad_types on ads.type = ad_types.id where cities.name = $1;';

$result = pg_query_params($dbconn, $query, array($city));

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($res_arr, $line);
}

$data = $res_arr;
header('Content-Type: application/json; charset=utf-8');
echo json_encode($data, JSON_UNESCAPED_UNICODE);
?>