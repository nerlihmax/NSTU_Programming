<?php

define('LOG_PATH', __DIR__ . '/log.txt');

function write_log($item_name, $quantity)
{
    $line = [
        'item' => $item_name,
        'quantity' => $quantity,
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

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST["encoding"])) {
    setcookie("encoding", $_POST["encoding"], $secs_in_year);
    header("location:index.php");
}

$item_name = trim($_POST['item_name']);
$quantity = trim($_POST['quantity']);

if (!empty($item_name) && !empty($quantity)) {
    write_log($item_name, $quantity);
    header('location:index.php');
}

?>

<html>

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <script src="https://cdn.tailwindcss.com"></script>
    <title>Лабораторная №1</title>
    <style>
        .body {
            @apply bg-slate-600
        }
    </style>
</head>

<body class="flex flex-col mx-auto h-full w-full justify-top items-center py-10 space-y-10">
    <div class="rounded-xl text-center p-10">
        <h2 class="text-2xl pb-3">Часть 1</h2>
        <a href="dump.sql">
            <h3 class="underline"><b>Скачать SQL дамп<b></h3>
        </a>
    </div>

    <div class="rounded-xl text-center">
        <h1 class="text-2xl font-normal">Теневая посылка</h1>

        <?php
        if (isset($_COOKIE["encoding"])) {
            echo "<div class=\"px-10 font-bold py-4\">Выбранная кодировка страницы: " . $_COOKIE["encoding"] . "</div>";
        }
        ?>

        <form method="post" action="index.php" class="flex flex-col space-y-6 text-base">
            <h1 class="text-xl font-normal">Выберите кодировку</h1>
            <div class="px-4 font-normal">
                <select name="encoding" class="space-y-4">
                    <option value="UTF-8" selected="selected">UTF-8</option>
                    <option value="UTF-16">UTF-16</option>
                    <option value="Unicode">Unicode</option>
                    <option value="cp-1251">cp-1251</option>
                </select>
            </div>
            <button class="py-6 w-full bg-gray-100 rounded-xl font-normal" type="submit">Отправить</button>
        </form>
    </div>


    <div class="rounded-xl text-center px-6 flex flex-col justify-center my-10 items-center space-y-6">
        <h1 class="text-2xl font-normal">Текстовый файл</h1>
        <div class="flex flex-col space-x-4 justify-center">
            <form method="post" action="index.php"
                class="flex flex-col space-y-6 text-base justify-center items-center">
                <h1 class="text-xl font-normal">Выберите наименование товара</h1>
                <div class="px-4 font-normal">
                    <select name="item_name" class="space-y-4">
                        <option value="Хлеб" selected="selected">Хлеб</option>
                        <option value="Молоко">Молоко</option>
                        <option value="Спички">Спички</option>
                        <option value="Масло">Масло</option>
                        <option value="Творог">Творог</option>
                    </select>
                </div>
                <div class="px-4 font-normal py-3 bg-gray-200">
                    <label for="quantity">Количество:</label>
                    <input type="number" name="quantity" required id="quantity">
                </div>

                <button class="py-6 w-full bg-gray-100 rounded-xl font-normal" type="submit">Отправить</button>
            </form>
        </div>
    </div>

    <div class="rounded-xl text-center px-6 flex flex-col mb-10 justify-center items-center">
        <h1 class="text-2xl pb-8 py-3">Журнал</h1>
        <?php
        $log = read_log();
        if (!empty($log)) {
            foreach ($log as $element) {
                $item = $element['item'];
                $quantity = $element['quantity'];
                ?>
                <div class="flex flex-col p-2 m-3 outline">
                    <p>
                        Товар:
                        <?= htmlspecialchars($item) ?>
                    </p>
                    <p>
                        Количество:
                        <?= htmlspecialchars($quantity) ?>
                    </p>
                </div>
                <?php
            }
        }
        ?>
    </div>
</body>

</html>