<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v8_labs');

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated && $_SESSION['group'] < 2) {
    header('location:forbidden.html');
    exit();
}

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$id = $queries["id"] or die("No id provided");

$result = pg_query_params($db, 'delete from teachers where id = $1;', array($id)) or die('Ошибка запроса: ' . pg_last_error());

pg_free_result($result);
pg_close($dbconn);

if ($result == true)
    header("location:index.php");
?>