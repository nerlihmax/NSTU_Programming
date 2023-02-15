<?php

require_once "config.php";

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$id = $queries["id"] or die("No id provided");
echo $id;

$dbconn = pg_connect("host=$host port=$port user=$user password=$password dbname=$db")
    or die('Не удалось соединиться: ' . pg_last_error());

$result = pg_query_params($dbconn, 'delete from ads where id = $1;', array($id)) or die('Ошибка запроса: ' . pg_last_error());

pg_free_result($result);
pg_close($dbconn);

if ($result == true)
    header("location:index.php");
?>