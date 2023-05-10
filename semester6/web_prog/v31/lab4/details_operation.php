<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v31');

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$worker = $queries["operation"] or die('Operation missing');

$query = 'SELECT 
        detail.id,
        operation.name as operation,
        detail.name as name,
        detail.duration
    from technological_map as detail
        inner join operation on operation.id = detail.operation
    where operation = $1 order by id;';

$result = pg_query_params($db, $query, array($worker));

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($res_arr, $line);
}

$data = $res_arr;
header('Content-Type: application/json; charset=utf-8');
echo json_encode($data, JSON_UNESCAPED_UNICODE);
?>