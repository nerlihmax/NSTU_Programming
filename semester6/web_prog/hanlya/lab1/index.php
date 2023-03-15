<?php

define('LOG_PATH', __DIR__ . '/log.txt');

function write_log($encoding, $font)
{
    $line = [
        'encoding' => $encoding,
        'font' => $font,
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

$timestamp = time();
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["time"])) {
    setcookie("timestamp", $timestamp, $secs_in_year);
    setcookie("time", $_POST["time"], $secs_in_year);
    header("location:index.php");
}

$encoding = trim($_POST['encoding']);
$font = trim($_POST['font']);

if (!empty($encoding) && !empty($font)) {
    write_log($encoding, $font);
    header('location:index.php');
}

?>

<html>

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Лабораторная №1</title>
    <style>
        .body {
            @apply bg-slate-600
        }
    </style>
</head>

<body class="flex flex-col mx-auto h-full w-full justify-top items-center bg-slate-600 py-10 space-y-10">
    <div class="bg-gray-100 rounded-xl text-center p-10">
        <h2 class="text-2xl pb-3">Часть 1</h2>
        <a href="dump.sql">
            <h3 class="underline"><b>Скачать SQL дамп<b></h3>
        </a>
    </div>

    <div class="bg-gray-100 rounded-xl text-center">
        <h1 class="text-2xl pb-8 py-3">Cookie</h1>

        <?php
        if (isset($_COOKIE["timestamp"]) && isset($_COOKIE["time"])) {
            echo "<div class=\"px-10\">Выбранный временной интервал: " . $_COOKIE["time"] . "</div>";
        }
        ?>

        <form method="post" action="index.php" class="flex flex-col space-y-6">
            <div class="flex flex-col space-y-4 pt-6">
                <div>
                    <input type="radio" name="time" value="8:30-9:00" id="1" checked="checked">
                    <label for="1">8:30-9:00</label>
                </div>

                <div>
                    <input type="radio" name="time" id="2" value="9:00-9:30">
                    <label for="2">8:30-9:00</label>
                </div>

                <div>
                    <input type="radio" name="time" value="9:30-10:00" id="3">
                    <label for="3">9:30-10:00</label>
                </div>

                <div>
                    <input type="radio" name="time" value="10:00-10:30" id="4">
                    <label for="4">10:00-10:30</label>
                </div>

                <div>
                    <input type="radio" name="time" value="10:30-11:00" id="5">
                    <label for="5">10:30-11:00</label>
                </div>

                <div>
                    <input type="radio" name="time" value="11:00-11:30" id="6">
                    <label for="6">11:00-11:30</label>
                </div>
            </div>
            <button class="p-6 bg-amber-400 rounded-xl" s type="submit">Отправить</button>
        </form>
    </div>

    <div class="bg-gray-100 rounded-xl text-center px-6 flex flex-col justify-center my-10 items-center space-y-6">
        <h1 class="text-2xl">Текстовый файл</h1>
        <div class="flex flex-col space-x-4 justify-center ">
            <h1 class="text-xl mb-4">Выберите шрифт:</h1>
            <form id="fontSelector" method="post" action="index.php" class="flex flex-col space-y-6 text-base">
                <select name="font" class="space-y-4">
                    <option value="Arial" selected="selected">Arial</option>
                    <option value="Times New Roman">Times New Roman</option>
                    <option value="Apple Mono">Apple Mono</option>
                </select>

                <h1 class="text-xl">Выберите кодировку</h1>

                <div class="flex flex-col space-y-4 pt-6">
                    <div>
                        <input type="radio" name="encoding" value="UTF-8" id="utf-8-select" checked="checked">
                        <label for="utf-8-select">UTF-8</label>
                    </div>

                    <div>
                        <input type="radio" name="encoding" id="utf-16-select" value="UTF-16">
                        <label for="utf-16-select">UTF-16</label>
                    </div>

                    <div>
                        <input type="radio" name="encoding" value="CP-1251" id="cp1251-select">
                        <label for="cp1251-select">CP-1251</label>
                    </div>
                </div>
                <button class="p-6 w-full bg-amber-400 rounded-xl" type="submit">Отправить</button>
            </form>
        </div>
    </div>

    <div class="bg-gray-100 rounded-xl text-center px-6 flex flex-col mb-10 justify-center items-center">
        <h1 class="text-2xl pb-8 py-3">Журнал</h1>
        <?php
        $log = read_log();
        if (!empty($log)) {
            foreach ($log as $element) {
                $encoding = $element['encoding'];
                $font = $element['font'];
                ?>
                <div class="flex flex-col p-2 m-3 outline-dotted">
                    <p>
                        Кодировка:
                        <?= htmlspecialchars($encoding) ?>
                    </p>
                    <p>
                        Шрифт:
                        <?= htmlspecialchars($font) ?>
                    </p>
                </div>
                <?php
            }
        }
        ?>
    </div>
</body>

</html>