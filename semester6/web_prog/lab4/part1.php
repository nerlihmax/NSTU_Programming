<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <title>Часть 1</title>
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.3/dist/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-validate/1.19.0/jquery.validate.min.js"></script>
</head>

<body>

    <h3>Валидация средствами JavaScript</h3>

    <form id="myForm" method="post" action="part1.php">
        <p><span>Имя туриста: </span><br><input id="tourist_name" type="text" name="tourist_name"></p>
        <p><span>Пожелания по формам отдыха: </span><br><textarea id="wishes" type="text" name="wishes"></textarea></p>
        <p><button type="submit">Отправить</button></p>
    </form>

    <script>
        $("#myForm").on("submit", (event) => {
            const isTouristNameValid = $("#tourist_name").val().length > 0
            const isWishesListValid = $("#wishes").val().length > 0
            if (!(isTouristNameValid && isWishesListValid)) {
                event.preventDefault()
                let res = ""
                if (!isTouristNameValid)
                    res += "Имя туриста не может быть пустым. "
                if (!isWishesListValid)
                    res += "Список пожеланий не может быть пустым."
                alert(res)
            }
        })
    </script>


    <h3>Валидация средствами JQuery</h3>

    <form id="myForm2" method="post" action="part1.php">
        <p><span>Имя туриста: </span><br><input id="tourist_name" type="text" name="tourist_name"></p>
        <p><span>Пожелания по формам отдыха: </span><br><textarea id="wishes" type="text" name="wishes"></textarea></p>
        <p><button type="submit">Отправить</button></p>
    </form>

    <script>
        $(document).ready(function () {
            $('#myForm2').validate({
                rules: {
                    tourist_name: {
                        required: true,
                    },
                    wishes: {
                        required: true,
                    }
                },
                messages: {
                    tourist_name: {
                        required: "Имя туриста не может быть пустым. ",
                    },
                    wishes: {
                        required: "Список пожеланий не может быть пустым.",
                    }
                },
            });
        });
    </script>


</body>

</html>