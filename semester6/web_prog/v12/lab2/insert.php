<?php
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v12');

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated || $_SESSION['group'] < 1) {
    header('location:forbidden.html');
    exit();
}

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
            'INSERT into documents
            (worker, name, date_of_apply, date_of_return) 
            values ($1, $2, $3, $4);',
            array(
                (int) $_POST['worker'],
                $_POST['document'],
                $_POST['apply_date'],
                $_POST['return_date'],
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
    <title>INSERT</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <form method="post" action="insert.php">
        <label for="worker">Исполнитель</label>
        <select id="worker" name="worker">
            <?php
            foreach ($workers as &$worker) {
                ?>
                <option value=<?= htmlspecialchars($worker["id"]) ?>><?= htmlspecialchars($worker["name"]) ?>
                    <?php
            }
            ?>
        </select>
        <br><br>
        <label for="document">Название документа</label>
        <input id="document" name="document" type="text" />
        <br><br>

        <label for="apply_date">Дата выдачи документа: </label>
        <input type="date" name="apply_date" id="apply_date">
        <br><br>

        <label for="return_date">Дата возврата документа: </label>
        <input type="date" name="return_date" id="return_date">
        <br><br>

        <p>
            <button type="submit">Добавить документ</button>
        </p>
    </form>
</body>