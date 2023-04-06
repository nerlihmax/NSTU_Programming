<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);
require_once(__DIR__ . '/../../utils/connection.php');
require_once(__DIR__ . '/../../utils/console_log.php');
$db = connect();

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated || $_SESSION['group'] < 1) {
    header('location:forbidden.html');
    exit();
}

$authors = pg_fetch_all(pg_query($db, 'SELECT DISTINCT id, name from author;'));
$publishers = pg_fetch_all(pg_query($db, 'SELECT id, name from publisher;'));
$sellers = pg_fetch_all(pg_query($db, 'SELECT id, name from seller;'));


if (
    $_SERVER['REQUEST_METHOD'] == 'POST'
    && isset($_POST["title"])
    && isset($_POST["author_id"])
    && isset($_POST["publisher_id"])
    && isset($_POST["seller_id"])
    && isset($_POST["year"])
    && isset($_POST["price"])
) {
    $result = pg_query_params(
        $db,
        'INSERT into book
            (title, author_id, publisher_id, seller_id, year, price) 
            values ($1, $2, $3, $4, $5, $6);',
        array(
            $_POST['title'],
            (int) $_POST['author_id'],
            (int) $_POST['publisher_id'],
            (int) $_POST['seller_id'],
            (int) $_POST['year'],
            (float) $_POST['price'],
        )
    );
    if ($result != false) {
        header("location:index.php#");
    }
}
?>

<html>

<head>
    <title>Table Insert</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <form method="post" action="insert.php">
        <label for="title">Название книги: </label>
        <input type="text" name="title" id="title">
        <br><br>

        <label for="year">Год издания: </label>
        <input type="number" name="year" id="year">
        <br><br>

        <label for="price">Цена в долларах: </label>
        <input type="number" name="price" id="price" step="0.01">
        <br><br>

        <label for="author">Автор</label>
        <select id="author" name="author_id">
            <?php
            foreach ($authors as &$author) {
                ?>
                <option value=<?= htmlspecialchars($author["id"]) ?>><?= htmlspecialchars($author["name"]) ?>
                    <?php
            }
            ?>
        </select>
        <br><br>
        <label for="publisher">Издатель</label>
        <select id="publisher" name="publisher_id">
            <?php
            foreach ($publishers as $publisher) {
                ?>
                <option value=<?= htmlspecialchars($publisher["id"]) ?>><?= htmlspecialchars($publisher["name"]) ?>
                    <?php
            }
            ?>
        </select>
        <br><br>

        <label for="seller">Магазин</label>
        <select id="seller" name="seller_id">
            <?php
            foreach ($sellers as $seller) {
                ?>
                <option value=<?= htmlspecialchars($seller["id"]) ?>><?= htmlspecialchars($seller["name"]) ?>
                    <?php
            }
            ?>
        </select>

        <p>
            <button type="submit">Создать запись</button>
        </p>
    </form>
</body>