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

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["text_styles"])) {
    $checked = $_POST['text_styles'];
    $result = '';
    for ($i = 0; $i < count($checked); $i++) {
        $result = $result . $checked[$i] . "\t";
    }
    setcookie("text_styles", $result, $secs_in_year);
    header("location:index.php");
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["styles"]) && isset($_POST["size"])) {
    $style_checked = $_POST['styles'];
    for ($i = 0; $i < count($style_checked); $i++) {
        $style = $style . $style_checked[$i] . "\t";
    }
    write_log($style, $_POST['size']);
    header('location:index.php');
}
?>

<html>

<head>
    <title>Лабораторная №1</title>
    <style>
        body {
            display: block;
            text-align: -webkit-center;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №1</h1>
    <h3>Вариант 12</h3>

    <h2>Часть 1</h2>
    <a href="dump.sql">
        <h3><b>SQL дамп<b></h3><br>
    </a>

    <br>
    <h2>Часть 2</h2>

    <div>
        <h2>Cookie</h2>

        <?php
        if (isset($_COOKIE["text_styles"])) {
            echo "<p>Выбранные стили текста: " . $_COOKIE["text_styles"] . "</p>";
        }
        ?>

        <h2>Выберите стили текста: </h2>
        <form method="post" action="index.php">
            <div>
                <input type="checkbox" name="text_styles[]" value="bold" /> Жирный<br />
                <input type="checkbox" name="text_styles[]" value="semibold" /> Полужирный<br />
                <input type="checkbox" name="text_styles[]" value="italic" /> Курсив<br />
            </div>
            <br>
            <button s type="submit">Отправить</button>
        </form>
    </div>

    <div>
        <h2>Текстовый файл</h2>
        <div>
            <form method="post" action="index.php">
                <h3>Выберите стиль приветствия:</h3>
                <div>
                    <input type="checkbox" name="styles[]" value="official" /> Официальное<br />
                    <input type="checkbox" name="styles[]" value="nonformal" /> Неофициальное<br />
                    <input type="checkbox" name="styles[]" value="basic" /> Простое<br />
                </div>

                <h3>Выберите размер: </h3>
                <select name="size">
                    <option value="10 pt" selected="selected">Маленький</option>
                    <option value="18 pt">Средний</option>
                    <option value="32 pt">Большой</option>
                </select>
                <button type="submit">Отправить</button>
            </form>
        </div>
    </div>

    <div>
        <h2>Журнал</h2>
        <?php
        $log = read_log();
        if (!empty($log)) {
            foreach ($log as $element) {
                $style = $element['style'];
                $size = $element['size'];
                ?>
                <div>
                    <p>
                        Вид приветствия:
                        <?= htmlspecialchars($style) ?>
                    </p>
                    <p>
                        Размер:
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