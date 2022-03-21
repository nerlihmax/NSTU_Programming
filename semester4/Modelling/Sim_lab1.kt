package simulation

data class Task(
    var time_task_arrived: Int = 0,
    var handle_time: Int = 0,
)

data class Computer(
    var work_ticks_left: Int = 0, //
)

val time_task_arrived_interval = (1..7)
val handle_time_interval = (1..12)

const val N = 1000 // Количество задач
const val iterations = 100 //  Количество итераций

var time_wait = 0 // Время ожидания
var time_idle_1 = 0 // Время простоя 1 машины
var time_idle_2 = 0 // Время простоя 2 машины
var tick_counter = 0 // Счётчик тактов
var n = 0 // Номер текущей задачи
var queue = 0 // Размер очереди
var completed_tasks_count = 0 // Счётчик выполненных задач

val tasks = Array(N + 1) { Task() } //список задач
val PCs = Array(2) { Computer() } //объекты для 2 компьютеров

val tasksQueue = Array(4) { Task() } //временная очередь для задач


fun setHandlerPC() {
    if (PCs[1].work_ticks_left == 0 && PCs[0].work_ticks_left != 0) { // Второй пк свободен и занят первый
        PCs[1].work_ticks_left = tasksQueue[queue].handle_time
        tasksQueue[queue] = Task(0, 0) //удаляем задачу из очереди
        queue--
    }
    if (PCs[0].work_ticks_left == 0) { // Если первый пк свободен
        PCs[0].work_ticks_left = tasksQueue[queue].handle_time
        tasksQueue[queue] = Task(0, 0)
        queue--
    }
    queue++
}

fun updateCounters() {
    if (PCs[1].work_ticks_left != 0 && PCs[0].work_ticks_left != 0 && queue != 0) time_wait += queue
    // Если обе машины заняты, и очередь не пуста - такт записывается в ожидание

    if (PCs[0].work_ticks_left == 0) time_idle_1++ //Если 1 пк не работает - такт записывается в простой 1 пк
    if (PCs[1].work_ticks_left == 0) time_idle_2++ // Аналогично для 2 пк
    if (PCs[0].work_ticks_left != 0) PCs[0].work_ticks_left-- //если 1 пк работал - вычитаем один такт из времени обработки задания
    if (PCs[1].work_ticks_left != 0) PCs[1].work_ticks_left-- // аналогично для 2 пк

}

fun main() {
    println("Starting program")
    var totalIdle1 = 0f //общее время простое для двух машин
    var totalIdle2 = 0f
    var totalAverageWait = 0f //общее время ожидания в очереди
    for (i in 0 until iterations) {
        tasks[0] = Task(time_task_arrived_interval.random(), handle_time_interval.random()) //создаём первую задачу
        for (j in 1 until N) { //создаем N-1 задач
            tasks[j] = Task(tasks[j - 1].time_task_arrived + time_task_arrived_interval.random(),
                handle_time_interval.random())
        }
        time_wait = 0 //время ожидания для конкретной итерации
        time_idle_1 = 0
        time_idle_2 = 0
        n = 0 //номер текущей задачи
        tick_counter = 1 //счетчик тактов
        completed_tasks_count = N //количество выполненных заданий
        while (true) {
            if (n == N && queue == 0 && PCs[0].work_ticks_left == 0 && PCs[1].work_ticks_left == 0) {
                break
                //если задач не осталось и оба компьютера освободились
            }
            updateCounters()
            if (tick_counter == tasks[n].time_task_arrived) {
                if (queue < 3) {
                    tasksQueue[queue] = tasks[n]
                    n++
                    setHandlerPC()
                } else {
                    tasks[n].handle_time = 0
                    completed_tasks_count--
                    n++
                }
            } else {
                if (queue != 0 && (PCs[0].work_ticks_left == 0 || PCs[1].work_ticks_left == 0)) {
                    queue--
                    setHandlerPC()
                }
            }
            tick_counter++
        }

        //считаем простой и время ожидания
        val avgWait = time_wait / N
        val idle1: Float = (time_idle_1.toFloat() / (tick_counter - 1).toFloat()) * 100.0f
        val idle2: Float = (time_idle_2.toFloat() / (tick_counter - 1).toFloat()) * 100.0f
        totalAverageWait += avgWait
        totalIdle1 += idle1
        totalIdle2 += idle2
        completed_tasks_count += N
        println("Процесс занял $tick_counter тиков")
        println("Выполнилось $N задач")
        println("Среднее время ожидания: $avgWait")
        println("Вероятность простоя 1 ПК: ${idle1.toString().substring(0, 4)}%")
        println("Вероятность простоя 2 ПК: ${idle2.toString().substring(0, 4)}%\n")
    }

    // находим среднее значение по итерациям
    totalAverageWait /= iterations
    totalIdle1 /= iterations
    totalIdle2 /= iterations
    completed_tasks_count /= iterations
    println("=====================================\n")
    println("В среднем выполнялось $completed_tasks_count задач в $iterations итерациях")
    println("Вероятность простоя 1 ПК: ${totalIdle1.toString().substring(0, 4)}%")
    println("Вероятность простоя 2 ПК: ${totalIdle2.toString().substring(0, 4)}%\n")
}