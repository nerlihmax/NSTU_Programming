<?php
ini_set('display_errors', '1');
ini_set('display_startup_errors', '1');
error_reporting(E_ALL);

require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v12');

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

$workers = pg_fetch_all(pg_query($db, 'SELECT id, name from workers;'));

if (
    $_SERVER['REQUEST_METHOD'] == 'POST'
    && isset($_POST["worker"])
    && isset($_POST["document"])
    && isset($_POST["apply_date"])
    && isset($_POST["return_date"])
) {
    if (strtotime($_POST['apply_date']) > strtotime($_POST['return_date'])) {
        echo '<script>alert("Дата возврата должна быть после даты выдачи");</script>';
    } else {
        $result = pg_query_params(
            $db,
            'UPDATE documents set
            worker = $1, name = $2, date_of_apply = $3, date_of_return = $4 where id = $5;',
            array(
                (int) $_POST['worker'],
                $_POST['document'],
                $_POST['apply_date'],
                $_POST['return_date'],
                $id,
            )
        );
        if ($result != false) {
            header("location:index.php#");
        }
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
        document.id, 
        worker.id as worker_id, 
        document.name as document, 
        document.date_of_apply, 
        document.date_of_return
    from documents as document
             inner join workers as worker on document.worker = worker.id
    WHERE document.id = $1';

    $result = pg_query_params($db, $query, array($id)) or die('Ошибка запроса: ' . pg_last_error());

    if ($result == false) {
        die("Failed to fetch row");
    }

    $row = pg_fetch_array($result, null, PGSQL_ASSOC);
    ?>

    <form method="post" action="update.php?id=<?php echo $id; ?>">
        <label for="worker">Фирма клиента</label>
        <select id="worker" name="worker">
            <?php
            foreach ($workers as &$worker) {
                ?>
                <option <?= ($worker["id"] == $row["worker_id"]) ? 'selected' : '' ?> value=<?= htmlspecialchars($worker["id"]) ?>>
                    <?= htmlspecialchars($worker["name"]) ?>
                    <?php
            }

            ?>
        </select>
        <br><br>

        <label for="document">Название документа</label>
        <input id="document" name="document" value="<?= $row["document"] ?>" />
        <br><br>

        <label for="apply_date">Дата выдачи документов: </label>
        <input type="date" name="apply_date" id="apply_date" value="<?= $row["date_of_apply"] ?>">
        <br><br>

        <label for="return_date">Дата возврата документа: </label>
        <input type="date" name="return_date" id="return_date" value="<?= $row["date_of_return"] ?>">
        <br><br>

        <p>
            <button type="submit">Редактировать запись</button>
        </p>
    </form>
</body>