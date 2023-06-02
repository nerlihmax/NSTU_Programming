package v3.presentation

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
import v3.data.Repository
import v3.data.entities.Department
import v3.data.entities.DisciplineSchedule
import v3.data.entities.Group
import v3.data.entities.asTableData
import v3.domain.entities.Teacher
import v3.domain.entities.TeacherDiscipline
import v3.domain.entities.asTableData
import java.time.LocalDate

class MainViewModel {
    private var database: Database = Database.connect(
        url = "jdbc:postgresql://localhost:32768/rgr_v3",
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = "postgrespw",
    )


    private val repository = Repository(database)

    sealed interface Route {
        object Departments : Route
        object Disciplines : Route
        object Teachers : Route
        object Groups : Route
        object DisciplinesSchedule : Route
    }

    sealed interface MainState {
        object Loading : MainState
        object Idle : MainState
        data class Editing(val row: Int) : MainState
        object Adding : MainState
        data class TeacherInfo(val data: List<TeacherDiscipline>) : MainState
        data class DisciplineInfo(val data: List<Teacher>) : MainState
    }

    sealed interface ErrorState {
        object Idle : ErrorState
        data class ShowError(val error: String) : ErrorState
    }

    private val _route: MutableStateFlow<Route> = MutableStateFlow(Route.Departments)
    val route: StateFlow<Route> = _route.asStateFlow()

    private val _state: MutableStateFlow<MainState> = MutableStateFlow(MainState.Loading)
    val state: StateFlow<MainState> = _state.asStateFlow()

    private val _errState: MutableStateFlow<ErrorState> = MutableStateFlow(ErrorState.Idle)
    val errState: StateFlow<ErrorState> = _errState.asStateFlow()

    private val _data: MutableStateFlow<TableData> = MutableStateFlow(TableData(emptyList(), emptyList()))
    val data: StateFlow<TableData> = _data.asStateFlow()

    init {
        loadData(this.route.value)
    }

    private fun loadData(route: Route) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { MainState.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route) {
                    Route.Departments -> {
                        val data = repository.Departments().getAll().asTableData
                        _data.update { data }
                    }

                    Route.Disciplines -> {
                        val data = repository.Disciplines().getAll().asTableData
                        _data.update { data }
                    }

                    Route.DisciplinesSchedule -> {
                        val data = repository.DisciplinesSchedule().getAll().asTableData
                        _data.update { data }
                    }

                    Route.Groups -> {
                        val data = repository.Groups().getAll().asTableData
                        _data.update { data }
                    }

                    Route.Teachers -> {
                        val data = repository.Teachers().getAll().asTableData
                        _data.update { data }
                    }
                }
            }
            awaitAll(job)
            _state.update { MainState.Idle }
        }
    }

    fun navigate(route: Route) {
        _route.update { route }
        loadData(route)
    }

    fun startEditing(id: Int) {
        _state.update { if (it is MainState.Editing) MainState.Idle else MainState.Editing(id) }
    }

    fun startAdding() {
        _state.update { if (it is MainState.Adding) MainState.Idle else MainState.Adding }
    }

    fun clearState() {
        _state.update { MainState.Idle }
        _errState.update { ErrorState.Idle }
    }

    fun hideInfo() {
        _state.update { MainState.Idle }
    }

    fun edit(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { MainState.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Route.Departments -> {
                        repository.Departments().update(v3.domain.entities.Department(row[0].toInt(), row[1]))
                    }

                    Route.Disciplines -> {
                        repository.Disciplines().update(v3.data.entities.Discipline {
                            id = row[0].toInt()
                            name = row[1]
                            semester = row[2].toInt()
                            specialty = row[3]
                        })
                    }

                    Route.DisciplinesSchedule -> {
                        repository.DisciplinesSchedule().update(DisciplineSchedule {
                            id = row[0].toInt()
                            discipline = repository.Disciplines().getByName(row[1]) ?: kotlin.run {
                                println("Discipline not found")
                                _errState.update { ErrorState.ShowError("Дисциплина не найдена") }
                                return@async
                            }
                            teacher = repository.Teachers().getByName(row[2]) ?: kotlin.run {
                                println("Teacher not found")
                                _errState.update { ErrorState.ShowError("Преподаватель не найден") }
                                return@async
                            }
                            hours = try {
                                row[3].toInt()
                            } catch (e: Exception) {
                                println("Bad format")
                                _errState.update { ErrorState.ShowError("Неправильный формат") }
                                return@async
                            }
                        })
                    }

                    Route.Groups -> {
                        repository.Groups().update(Group {
                            id = row[0].toInt()
                            name = row[1]
                            specialty = row[2]
                        })
                    }

                    Route.Teachers -> {
                        repository
                            .Teachers()
                            .update(Teacher(
                                id = row[0].toInt(),
                                fullName = row[1],
                                department = repository.Departments().getByName(row[2]) ?: kotlin.run {
                                    println("Department not found")
                                    _errState.update { ErrorState.ShowError("Отдел не найден") }
                                    return@async
                                },
                                post = row[3],
                                hireDate = try {
                                    LocalDate.parse(row[4]).also {
                                        if (it > LocalDate.now()) {
                                            println("Wrong date format")
                                            _errState.update { ErrorState.ShowError("Неправильный формат даты") }
                                            return@async
                                        }
                                    }
                                } catch (e: Exception) {
                                    println("Wrong date format")
                                    _errState.update { ErrorState.ShowError("Неправильный формат даты") }
                                    return@async
                                },
                            ))
                    }
                }
            }
            awaitAll(job)
            _state.update { MainState.Idle }
            loadData(route.value)
        }
    }


    fun deleteRow(id: Int) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { MainState.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Route.Departments -> {
                        repository.Departments().delete(id)
                    }

                    Route.Disciplines -> {
                        repository.Disciplines().delete(id)
                    }

                    Route.DisciplinesSchedule -> {
                        repository.DisciplinesSchedule().delete(id)
                    }

                    Route.Groups -> {
                        repository.Groups().delete(id)
                    }

                    Route.Teachers -> {
                        repository.Teachers().delete(id)
                    }
                }
            }
            awaitAll(job)
            _state.update { MainState.Idle }
            loadData(route.value)
        }
    }

    fun showInfo(id: Int) {
        when (route.value) {
            Route.Teachers -> {
                val info = repository.getTeacherInfo(id)
                _state.update { MainState.TeacherInfo(info) }
            }

            Route.Disciplines -> {
                val discipline = repository.getTeachersByDiscipline(id)
                _state.update { MainState.DisciplineInfo(discipline) }
            }

            else -> {
                println("Not implemented")
            }
        }
    }

    fun add(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { MainState.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route.value) {
                    Route.Departments -> {
                        repository.Departments().create(Department {
                            name = row[1]
                        })
                    }

                    Route.Disciplines -> {
                        repository.Disciplines().create(v3.data.entities.Discipline {
                            name = row[1]
                            semester = row[2].toInt()
                            specialty = row[3]
                        })
                    }

                    Route.DisciplinesSchedule -> {
                        repository.DisciplinesSchedule().create(DisciplineSchedule {
                            discipline = repository.Disciplines().getByName(row[1]) ?: kotlin.run {
                                println("Discipline not found")
                                _errState.update { ErrorState.ShowError("Дисциплина не найдена") }
                                return@async
                            }
                            teacher = repository.Teachers().getByName(row[2]) ?: kotlin.run {
                                println("Teacher not found")
                                _errState.update { ErrorState.ShowError("Преподаватель не найден") }
                                return@async
                            }
                            hours = try {
                                row[3].toInt()
                            } catch (e: Exception) {
                                println("Bad format")
                                _errState.update { ErrorState.ShowError("Неправильный формат") }
                                return@async
                            }
                        })
                    }

                    Route.Groups -> {
                        repository.Groups().create(Group {
                            name = row[1]
                            specialty = row[2]
                        })
                    }

                    Route.Teachers -> {
                        repository.Teachers().create(v3.data.entities.Teacher {
                            fullName = row[1]
                            department = repository.Departments().getByName(row[2]) ?: kotlin.run {
                                println("Department not found")
                                _errState.update {
                                    ErrorState.ShowError(
                                        "Отдел не найден, проверьте опечатки"
                                    )
                                }
                                return@async
                            }
                            post = row[3]
                            hireDate = try {
                                LocalDate.parse(row[4]).also {
                                    if (it > LocalDate.now()) {
                                        println("Wrong date format")
                                        _errState.update { ErrorState.ShowError("Неправильный формат даты") }
                                        return@async
                                    }
                                }
                            } catch (e: Exception) {
                                println("Wrong date format")
                                _errState.update { ErrorState.ShowError("Неправильный формат даты") }
                                return@async
                            }
                        })
                    }
                }
            }
            awaitAll(job)
            _state.update { MainState.Idle }
            loadData(route.value)
        }
    }
}