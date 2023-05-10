<?php
require_once(__DIR__ . '/../../utils/connection.php');
$db = connect('web_v31');

session_start();

$authenticated = !empty($_SESSION['auth']);
$authenticated = $authenticated && $_SESSION['auth'] == true;

if (!$authenticated || $_SESSION['group'] < 1) {
    header('location:no_access.html');
    exit();
}

$operations = pg_fetch_all(pg_query($db, 'SELECT id, name from operation;'));

if (
    $_SERVER['REQUEST_METHOD'] == 'POST'
    && isset($_POST["operation"])
    && isset($_POST["name"])
    && isset($_POST["duration"])
) {
    $result = pg_query_params(
        $db,
        'INSERT into technological_map
            (name, operation, duration) 
            values ($1, $2, $3);',
        array(
            $_POST['name'],
            (int) $_POST['operation'],
            (int) $_POST['duration'],
        )
    );
    if ($result != false) {
        header("location:index.php#");

    }
}
?>

<html>

<head>
    <title>Создание записи</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <form method="post" action="insert.php">
        <label for="operation">Вид операции</label>
        <select id="operation" name="operation">
            <?php
            foreach ($operations as &$operation) {
                ?>
                <option value=<?= htmlspecialchars($operation["id"]) ?>><?= htmlspecialchars($operation["name"]) ?>
                    <?php
            }
            ?>
        </select>
        <br><br>
        <label for="name">Название детали</label>
        <input id="name" name="name" type="text" />
        <br><br>

        <label for="duration">Длительность обработки (минуты): </label>
        <input type="number" name="duration" id="duration" min=10>
        <br><br>
        <p>
            <button type="submit">Создать запись</button>
        </p>
    </form>
</body>