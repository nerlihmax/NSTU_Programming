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
    <title>Изменение данных</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <?php

    $queries = array();
    parse_str($_SERVER["QUERY_STRING"], $queries);

    $id = $queries["id"] or die("No id provided");

    if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["reader"]) && isset($_POST["name"]) && isset($_POST["date_of_issue"]) && isset($_POST["date_of_return"])) {
        $reader_id = pg_fetch_all(pg_query_params($db, 'select id from reader where name = $1 limit 1;', array($_POST['reader'])), PGSQL_ASSOC)[0];
        if (!$reader_id['id'] || $reader_id['id'] == 0) {
            pg_query_params($db, 'insert into reader (name) values ($1);', array($_POST['reader']));
            $reader_id = pg_fetch_assoc(pg_query_params($db, 'select id from reader where name = $1 limit 1;', array($_POST['reader'])));
        }

        if ($_POST['date_of_return'] && (strtotime($_POST['date_of_issue']) > strtotime($_POST['date_of_return']))) {
            echo '<script>alert("Дата возврата должна быть после даты выдачи");</script>';
        } else {
            $result = pg_query_params(
                $db,
                'update issued_books set
                (name, reader, date_of_issue, date_of_return) = ($1, $2, $3, $4) where id = $5',
                array(
                    $_POST['name'],
                    (int) $reader_id['id'],
                    $_POST['date_of_issue'],
                    (isset($_POST['date_of_return']) ? $_POST['date_of_return'] : 'null'),
                    $id
                )
            );
            if ($result != false) {
                header("location:index.php");
            }
        }
    }

    $query = '
    select book.id,
    book.name,
    reader.name as reader_name,
    book.date_of_issue,
    book.date_of_return
    from issued_books as book
    inner join reader on reader.id = book.reader 
    where book.id = ' . $id;

    $result = pg_query($db, $query) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        echo "Failed to fetch row";
        header("location:index.php");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);
    ?>

    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <p><span>Читатель: </span><input type="text" name="reader" value="<?= $row['reader_name'] ?>"></p>
        <p><span>Название книги: </span><input type="text" name="name" value="<?= $row['name'] ?>"></p>
        <p><span>Дата получения: </span><input type="date" name="date_of_issue" value="<?= $row['date_of_issue'] ?>">
        </p>
        <p><span>Дата возврата: </span><input type="date" name="date_of_return" value="<?= $row['date_of_return'] ?>">
        </p>
        <p><button type="submit">Обновить</button></p>
    </form>
</body>