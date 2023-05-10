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
    header("location:part1.php");
}

$size = trim($_POST['size']);
$style = trim($_POST['style']);

if (!empty($size) && !empty($style)) {
    write_log($style, $size);
    header('location:part1.php');
}

?>

<html>

<head>
    <title>Лабораторная №4</title>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-validate/1.19.0/jquery.validate.min.js"></script>
    <style>
        .error {
            color: red;
        }
    </style>
</head>

<body>
    <h1>Лабораторная №4</h1>
    <h3>Вариант 11</h3>

    <div>
        <h2>Валидация средствами js</h2>

        <?php
        if (isset($_COOKIE["font"])) {
            echo "<div>Выбранный шрифт: " . $_COOKIE["font"] . "</div>";
        }
        ?>
        <br>
        <form method="post" id="form1" action="part1.php">
            <div>
                <div>
                    <input type="radio" name="font" value="Arial" id="1">
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
        <script>
            function radiobuttons(name) {
                var radios = document.getElementsByName(name);
                var formValid = false;

                var i = 0;
                while (!formValid && i < radios.length) {
                    if (radios[i].checked) formValid = true;
                    i++;
                }
                return formValid;
            }

            $("#form1").on("submit", (event) => {
                const isradiobuttonsValid = radiobuttons("font")
                if (!isradiobuttonsValid) {
                    event.preventDefault()
                    const res = "Шрифт не выбран"
                    alert(res)
                }
            })
        </script>
    </div>

    <div>
        <h2>Валидация Jquery</h2>
        <div>
            <form method="post" id="form2" action="part1.php">
                <h3>Выберите стиль приветствия:</h3>
                <div id="styles">
                    <div>
                        <input type="radio" name="style" value="Официальное" id="21">
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
                    <option value="" selected disabled hidden>%Размер шрифта%</option>
                    <option value="10 pt">Маленький (10pt)</option>
                    <option value="18 pt">Средний (18pt)</option>
                    <option value="32 pt">Большой (32pt)</option>
                </select>
                <br>
                <button type="submit">Отправить</button>
            </form>
            <script>
                $(document).ready(function () {
                    $("#form2").validate({
                        rules: {
                            style: {
                                required: true,
                            },
                            size: {
                                required: true,
                            }
                        },
                        messages: {
                            style: {
                                required: "Стиль приветствия не выбраны.",
                            },
                            size: {
                                required: "Размер шрифта не выбран. ",
                            }
                        },
                        errorElement: "div",
                        errorPlacement: function (error, element) {
                            if (element.is(":radio")) {
                                error.appendTo(element.parents('#styles'));
                            }
                            else {
                                error.insertAfter(element);
                            }
                        },
                    })
                });
            </script>
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