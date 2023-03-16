<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$reader = $queries["reader"] or die('Reader ID required');

$query = '
    select book.id,
    book.name,
    reader.name as reader_name,
    book.date_of_issue,
    book.date_of_return
    from issued_books as book
    inner join reader on reader.id = book.reader 
    where book.reader = $1';



$result = pg_query_params($db, $query, array($reader));

$res_arr = array();

while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
    array_push($res_arr, $line);
}

$data = $res_arr;
header('Content-Type: application/json; charset=utf-8');
echo json_encode($data, JSON_UNESCAPED_UNICODE);
?>