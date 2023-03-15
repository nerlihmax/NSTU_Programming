<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect();

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated && $_SESSION['group'] < 2) {
    header('location:forbidden.html');
    exit();
}

?>

<html>

<head>
    <title>Вставка в таблицу</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>

    <?php
    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["reader"]) && isset($_POST["name"]) && isset($_POST["date_of_issue"]) && isset($_POST["date_of_return"])) {
        $reader_id = pg_fetch_assoc(pg_query_params($db, 'select id from reader where name = $1', array($_POST['reader'])), PGSQL_ASSOC);

        if (!isset($reader_id['id']) || $reader_id['id'] == 0) {
            pg_query_params($db, 'insert into reader (name) values ($1)', array($_POST['reader']));
            $reader_id = pg_query_params($db, 'select id from reader where name = $1', array($_POST['reader']));
        }

        if (strtotime($_POST['date_of_issue']) > strtotime($_POST['date_of_return'])) {
            echo '<script>alert("Дата возврата должна быть после даты выдачи");</script>';
        } else {
            $result = pg_query_params(
                $db,
                'insert into issued_books
                (name, reader, date_of_issue, date_of_return) values ($1, $2, $3, $4);',
                array(
                    $_POST['name'],
                    (int) $reader_id['id'],
                    $_POST['date_of_issue'],
                    (isset($_POST['date_of_return']) ? $_POST['date_of_return'] : 'null'),
                )
            );
            if ($result != false) {
                header("location:index.php");
            }
        }
    }
    ?>


    <form method="post" action="insert.php">
        <p><span>Читатель: </span><input type="text" name="reader"></p>
        <p><span>Название книги: </span><input type="text" name="name"></p>
        <p><span>Дата получения: </span><input type="date" name="date_of_issue"></p>
        <p><span>Дата возврата: </span><input type="date" name="date_of_return"></p>
        <p><button type="submit">Создать запись</button></p>
    </form>
</body>