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

$authors = pg_fetch_all(pg_query($db, 'SELECT DISTINCT id, name from author;'));
$publishers = pg_fetch_all(pg_query($db, 'SELECT id, name from publisher;'));
$sellers = pg_fetch_all(pg_query($db, 'SELECT id, name from seller;'));

$queries = array();
parse_str($_SERVER["QUERY_STRING"], $queries);

$id = $queries["id"] or die("No id provided");

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
        'UPDATE book set
        (title, author_id, publisher_id, seller_id, year, price) 
            = ($1, $2, $3, $4, $5, $6) where id = $7;',
        array(
            $_POST['title'],
            (int) $_POST['author_id'],
            (int) $_POST['publisher_id'],
            (int) $_POST['seller_id'],
            (int) $_POST['year'],
            (float) $_POST['price'],
            $id,
        )
    );
    if ($result != false) {
        header("location:index.php#");
    }
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

    $query = 'SELECT 
        book.id, 
        book.title,
        book.year,
        book.price,
        a.id    AS author,
        p.id    AS publisher,
        s.id    AS seller
    FROM book
        INNER JOIN author a ON a.id = book.author_id
        INNER JOIN publisher p ON book.publisher_id = p.id
        INNER JOIN seller s ON s.id = book.seller_id
    WHERE book.id = ' . $id;

    $result = pg_query($db, $query) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        echo "Failed to fetch row";
        header("location:index.php");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);
    ?>

    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <label for="title">Название книги: </label>
        <input type="text" name="title" id="title" value="<?= $row["title"] ?>">
        <br><br>

        <label for="year">Год издания: </label>
        <input type="number" name="year" id="year" value="<?= $row["year"] ?>">
        <br><br>

        <label for="price">Цена в долларах: </label>
        <input type="number" name="price" id="price" step="0.01" value="<?= $row["price"] ?>">
        <br><br>

        <label for="author">Автор</label>
        <select id="author" name="author_id">
            <?php
            foreach ($authors as &$author) {
                ?>
                <option <?= ($author["id"] == $row["author"]) ? 'selected' : '' ?> value=<?= htmlspecialchars($author["id"]) ?>>
                    <?= htmlspecialchars($author["name"]) ?>
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
                <option <?= ($publisher["id"] == $row["publisher"]) ? 'selected' : '' ?>
                    value=<?= htmlspecialchars($publisher["id"]) ?>><?= htmlspecialchars($publisher["name"]) ?>
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
                <option <?= ($seller["id"] == $row["seller"]) ? 'selected' : '' ?> value=<?= htmlspecialchars($seller["id"]) ?>>
                    <?= htmlspecialchars($seller["name"]) ?>
                    <?php
            }
            ?>
        </select>

        <p>
            <button type="submit">Сохранить изменения</button>
        </p>
    </form>
</body>