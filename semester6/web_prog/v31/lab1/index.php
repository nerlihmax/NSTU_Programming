<?php

define('LOG_PATH', __DIR__ . '/log.txt');

function write_log($style, $size)
{
    $line = [
        'style' => $style,
        'size' => $size,
    ];
    $json = json_encode($line);
    file_put_contents(LOG_PATH, "$json\n", FILE_APPEND);
}

function read_log()
{
    $log = [];
    $json_lines = file_get_contents(LOG_PATH);

    foreach (explode("\n", $json_lines) as $line) {
        if (empty($line))
            continue;
        $log[] = json_decode($line, true);
    }

    return $log;
}

$secs_in_year = time() + 60 * 60 * 24 * 365;

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["font"])) {
    setcookie("font", $_POST["font"], $secs_in_year);
    header("location:index.php");
}

$size = trim($_POST['size']);
$style = trim($_POST['style']);

if (!empty($size) && !empty($style)) {
    write_log($style, $size);
    header('location:index.php');
}

?>

<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Лабораторная №1</title>
    <style>
        #sourceLink {
            font-size: x-large;
        }
    </style>
</head>

<body>
    <a href="src.html" id="sourceLink">Исходный код</a>
    <div>
        <h2>Часть 1</h2>
        <a href="dump.sql">
            <h3><b>SQL dump<b></h3>
        </a>
    </div>

    <div>
        <h1>Cookie</h1>

        <?php
        if (isset($_COOKIE["font"])) {
            echo "<div>Выбранный шрифт: " . $_COOKIE["font"] . "</div>";
        }
        ?>
        <br>
        <form method="post" action="index.php">
            <div>
                <div>
                    <input type="radio" name="font" value="Arial" id="1" checked="checked">
                    <label for="1">Arial</label>
                </div>

                <div>
                    <input type="radio" name="font" id="2" value="Times New Roman">
                    <label for="2">Times New Roman</label>
                </div>

                <div>
                    <input type="radio" name="font" value="Roboto" id="3">
                    <label for="3">Roboto</label>
                </div>

                <div>
                    <input type="radio" name="font" value="Colibri" id="4">
                    <label for="4">Colibri</label>
                </div>

                <div>
                    <input type="radio" name="font" value="Courier New" id="5">
                    <label for="5">Courier New</label>
                </div>
            </div>
            <br>
            <button s type="submit">Отправить</button>
        </form>
    </div>

    <div>
        <h1>Текстовый файл</h1>
        <div>
            <form method="post" action="index.php">
                <h3>Выберите стиль приветствия:</h3>
                <div>
                    <div>
                        <input type="radio" name="style" value="Официальное" id="21" checked="checked">
                        <label for="21">Официальное</label>
                    </div>

                    <div>
                        <input type="radio" name="style" id="22" value="Неофициальное">
                        <label for="22">Неофициальное</label>
                    </div>

                    <div>
                        <input type="radio" name="style" value="Дружеское" id="23">
                        <label for="23">Дружеское</label>
                    </div>
                </div>

                <h3>Выберите размер шрифта: </h3>
                <select name="size">
                    <option value="10 pt" selected="selected">Маленький (10pt)</option>
                    <option value="18 pt">Средний (18pt)</option>
                    <option value="32 pt">Большой (32pt)</option>
                </select>
                <button type="submit">Отправить</button>
            </form>
        </div>
    </div>

    <div>
        <h1>Журнал</h1>
        <?php
        $log = read_log();
        if (!empty($log)) {
            foreach ($log as $element) {
                $style = $element['style'];
                $size = $element['size'];
                ?>
                <div>
                    <p>
                        Стиль приветствия:
                        <?= htmlspecialchars($style) ?>
                    </p>
                    <p>
                        Размер шрифта:
                        <?= htmlspecialchars($size) ?>
                    </p>
                </div>
                <?php
            }
        }
        ?>
    </div>
</body>

</html>