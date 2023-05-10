<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$position = $queries["degree"] or die('Input data required');

$query = 'SELECT teachers.id,
position.name as position,
degree.name as degree,
courses.name as courses,
teachers.surname as teacher,
teachers.room_number
from teachers as teachers
inner join position on position.id = teachers.position
inner join degree on degree.id = teachers.degree
inner join courses on courses.id = teachers.courses
where teachers.degree = $1';

$result = pg_query_params($db, $query, array($position));

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($res_arr, $line);
}

$data = $res_arr;
header('Content-Type: application/json; charset=utf-8');
echo json_encode($data, JSON_UNESCAPED_UNICODE);
?>