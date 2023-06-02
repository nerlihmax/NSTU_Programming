package v7.presentation

import core.TableData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.ktorm.database.Database
import v7.data.Repository
import v7.data.entities.Course
import v7.data.entities.CourseCompletion
import v7.data.entities.Department
import v7.data.entities.Employee
import v7.data.entities.Position
import v7.data.entities.asTableData
import v7.presentation.state_holders.ErrorStates
import v7.presentation.state_holders.Routes
import v7.presentation.state_holders.State
import java.time.LocalDate

class ViewModel {
    private var database: Database = Database.connect(
        url = "jdbc:postgresql://localhost:32768/rgr_v7",
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = "postgrespw",
    )

    private val repository = Repository(database)

    private val route: MutableStateFlow<Routes> = MutableStateFlow(Routes.Departments)

    private val _state: MutableStateFlow<State> = MutableStateFlow(State.Loading)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _errState: MutableStateFlow<ErrorStates> = MutableStateFlow(ErrorStates.Idle)
    val errState: StateFlow<ErrorStates> = _errState.asStateFlow()

    private val _data: MutableStateFlow<TableData> = MutableStateFlow(TableData(emptyList(), emptyList()))
    val data: StateFlow<TableData> = _data.asStateFlow()

    init {
        loadData(this.route.value)
    }

    private fun loadData(route: Routes) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route) {
                    Routes.Departments -> {
                        val data = repository.Departments().getAll().asTableData
                        _data.update { data }
                    }

                    Routes.Courses -> {
                        val data = repository.Courses().getAll().asTableData
                        _data.update { data }
                    }

                    Routes.CoursesCompletion -> {
                        val data = repository.CoursesCompletions().getAll().asTableData
                        _data.update { data }
                    }

                    Routes.Employees -> {
                        val data = repository.Employees().getAll().asTableData
                        _data.update { data }
                    }

                    Routes.Positions -> {
                        val data = repository.Positions().getAll().asTableData
                        _data.update { data }
                    }
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
        }
    }

    fun navigate(route: Routes) {
        this.route.update { route }
        loadData(route)
    }

    fun startEditing(id: Int) {
        _state.update { if (it is State.Editing) State.Idle else State.Editing(id) }
    }

    fun startAdding() {
        _state.update { if (it is State.Adding) State.Idle else State.Adding }
    }

    fun clearState() {
        _state.update { State.Idle }
        _errState.update { ErrorStates.Idle }
    }

    fun hideInfo() {
        _state.update { State.Idle }
    }

    fun edit(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Routes.Departments -> {
                        repository.Departments().update(Department {
                            id = row[0].toInt()
                            name = row[1]
                        })
                    }

                    Routes.Courses -> {
                        repository.Courses().update(Course {
                            id = row[0].toInt()
                            name = row[1]
                            department = repository.Departments().getByName(row[2]) ?: run {
                                _errState.update { ErrorStates.ShowError("Отдел с именем ${row[2]} не найден!") }
                                return@async
                            }
                            hours = row[3].toInt()
                            description = row[4]
                        })
                    }

                    Routes.CoursesCompletion -> {
                        repository.CoursesCompletions().update(CourseCompletion {
                            id = row[0].toInt()
                            employee = repository.Employees().getByName(row[1]) ?: run {
                                _errState.update { ErrorStates.ShowError("Сотрудник с именем ${row[1]} не найден!") }
                                return@async
                            }
                            course = repository.Courses().getByName(row[2]) ?: run {
                                _errState.update { ErrorStates.ShowError("Курс с именем ${row[2]} не найден!") }
                                return@async
                            }
                            startDate = try {
                                LocalDate.parse(row[3]).also {
                                    if (it > LocalDate.now()) {
                                        println("Wrong date format")
                                        _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                        return@async
                                    }
                                }
                            } catch (e: Exception) {
                                println("Wrong date format")
                                _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                return@async
                            }
                        })
                    }

                    Routes.Employees -> {
                        repository.Employees().update(Employee {
                            id = row[0].toInt()
                            name = row[1]
                            surname = row[2]
                            department = repository.Departments().getByName(row[3]) ?: run {
                                _errState.update { ErrorStates.ShowError("Отдел с именем ${row[3]} не найден!") }
                                return@async
                            }
                            position = repository.Positions().getByName(row[4]) ?: run {
                                _errState.update { ErrorStates.ShowError("Должность с именем ${row[4]} не найдена!") }
                                return@async
                            }
                            hireDate = try {
                                LocalDate.parse(row[5]).also {
                                    if (it > LocalDate.now()) {
                                        _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                        return@async
                                    }
                                }
                            } catch (e: Exception) {
                                _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                return@async
                            }
                        })
                    }

                    Routes.Positions -> {
                        repository.Positions().update(Position {
                            id = row[0].toInt()
                            name = row[1]
                        })
                    }
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(route.value)
        }
    }


    fun deleteRow(id: Int) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Routes.Departments -> repository.Departments().delete(id)
                    Routes.Courses -> repository.Courses().delete(id)
                    Routes.CoursesCompletion -> repository.CoursesCompletions().delete(id)
                    Routes.Employees -> repository.Employees().delete(id)
                    Routes.Positions -> repository.Positions().delete(id)
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(route.value)
        }
    }

    fun showInfo(id: Int) {
        when (route.value) {
            else -> {
                println("Not implemented")
            }
        }
    }

    fun add(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Routes.Departments -> {
                        repository.Departments().create(Department {
                            name = row[1]
                        })
                    }

                    Routes.Courses -> repository.Courses().create(Course {
                        name = row[1]
                        department = repository.Departments().getByName(row[2]) ?: run {
                            _errState.update { ErrorStates.ShowError("Отдел с именем ${row[2]} не найден!") }
                            return@async
                        }
                        hours = row[3].toInt()
                        description = row[4]
                    })


                    Routes.CoursesCompletion -> repository.CoursesCompletions().create(CourseCompletion {
                        course = repository.Courses().getByName(row[1]) ?: run {
                            _errState.update { ErrorStates.ShowError("Курс с именем ${row[1]} не найден!") }
                            return@async
                        }
                        employee = repository.Employees().getByName(row[2]) ?: run {
                            _errState.update { ErrorStates.ShowError("Сотрудник с именем ${row[2]} не найден!") }
                            return@async
                        }
                        startDate = try {
                            LocalDate.parse(row[3]).also {
                                if (it > LocalDate.now()) {
                                    println("Wrong date format")
                                    _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                    return@async
                                }
                            }
                        } catch (e: Exception) {
                            println("Wrong date format")
                            _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                            return@async
                        }
                    })

                    Routes.Employees -> repository.Employees().create(Employee {
                        name = row[1]
                        surname = row[2]
                        department = repository.Departments().getByName(row[3]) ?: run {
                            _errState.update { ErrorStates.ShowError("Отдел с именем ${row[3]} не найден!") }
                            return@async
                        }
                        position = repository.Positions().getByName(row[4]) ?: run {
                            _errState.update { ErrorStates.ShowError("Должность с именем ${row[4]} не найдена!") }
                            return@async
                        }
                        hireDate = try {
                            LocalDate.parse(row[5]).also {
                                if (it > LocalDate.now()) {
                                    _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                                    return@async
                                }
                            }
                        } catch (e: Exception) {
                            _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                            return@async
                        }
                    })

                    Routes.Positions -> repository.Positions().create(Position {
                        name = row[1]
                    })
                }
            }

            awaitAll(job)
            _state.update { State.Idle }
            loadData(route.value)
        }
    }
}