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
        url = "jdbc:postgresql://localhost:32768/postgres",
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = "postgrespw",
    )

    private val repository = Repository(database)

    private val _route: MutableStateFlow<Routes> = MutableStateFlow(Routes.Departments)
    val route: StateFlow<Routes> = _route.asStateFlow()

    private val _state: MutableStateFlow<State> = MutableStateFlow(State.Loading)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _errState: MutableStateFlow<ErrorStates> = MutableStateFlow(ErrorStates.Idle)
    val errState: StateFlow<ErrorStates> = _errState.asStateFlow()

    private val _data: MutableStateFlow<TableData> = MutableStateFlow(TableData(emptyList(), emptyList()))
    val data: StateFlow<TableData> = _data.asStateFlow()

    init {
        loadData(this._route.value)
    }

    private fun loadData(route: Routes) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route) {
                    Routes.Departments -> _data.update { repository.Departments().getAll().asTableData }
                    Routes.Courses -> _data.update { repository.Courses().getAll().asTableData }
                    Routes.CoursesCompletion -> _data.update { repository.CoursesCompletions().getAll().asTableData }
                    Routes.Employees -> _data.update { repository.Employees().getAll().asTableData }
                    Routes.Positions -> _data.update { repository.Positions().getAll().asTableData }
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
        }
    }

    fun navigate(route: Routes) {
        this._route.update { route }
        loadData(route)
    }

    fun startEditing(id: Int) = _state.update { if (it is State.Editing) State.Idle else State.Editing(id) }

    fun startAdding() = _state.update { if (it is State.Adding) State.Idle else State.Adding }

    fun clearState() {
        _state.update { State.Idle }
        _errState.update { ErrorStates.Idle }
    }


    fun add(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Departments -> repository.Departments().create(updateData(_route.value, row))
                    Routes.Courses -> repository.Courses().create(updateData(_route.value, row))
                    Routes.CoursesCompletion -> repository.CoursesCompletions().create(updateData(_route.value, row))
                    Routes.Employees -> repository.Employees().create(updateData(_route.value, row))
                    Routes.Positions -> repository.Positions().create(updateData(_route.value, row))
                }
            }

            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    fun edit(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Departments -> repository.Departments().update(
                        updateData(_route.value, row)
                    )

                    Routes.Courses -> repository.Courses().update(updateData(_route.value, row))
                    Routes.CoursesCompletion -> repository.CoursesCompletions().update(updateData(_route.value, row))
                    Routes.Employees -> repository.Employees().update(updateData(_route.value, row))
                    Routes.Positions -> repository.Positions().update(updateData(_route.value, row))
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    fun deleteRow(id: Int) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Departments -> repository.Departments().delete(id)
                    Routes.Courses -> repository.Courses().delete(id)
                    Routes.CoursesCompletion -> repository.CoursesCompletions().delete(id)
                    Routes.Employees -> repository.Employees().delete(id)
                    Routes.Positions -> repository.Positions().delete(id)
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    fun showCurrentCourses() {
        val courses = repository.getEmployeesCurrentCourses(LocalDate.now().monthValue, LocalDate.now().year)
        _state.update { State.ShowCurrentCourses(courses) }
    }

    fun showDepartmentCourses(id: Int) {
        val courses = repository.getPassedEmployeeCoursesByDepartment(id)
        _state.update { State.ShowPassedCourses(courses) }
    }

    fun showPlannedCourses() {
        val courses = repository.getEmployeesPlannedCourses()
        _state.update { State.ShowPlannedCourses(courses) }
    }

    fun hideInfo() = _state.update { State.Idle }

    @Suppress("UNCHECKED_CAST")
    private fun <T> updateData(route: Routes, row: List<String>): T = when (route) {
        Routes.Courses -> Course {
            if (row[0].isNotBlank()) id = row[0].toInt()
            name = row[1]
            department = repository.Departments().getByName(row[2]) ?: run {
                _errState.update { ErrorStates.ShowError("Отдел с именем ${row[2]} не найден!") }
                return@Course
            }
            hours = row[3].toInt()
            description = row[4]
        } as T

        Routes.CoursesCompletion -> CourseCompletion {
            if (row[0].isNotBlank()) id = row[0].toInt()
            course = repository.Courses().getByName(row[1]) ?: run {
                _errState.update { ErrorStates.ShowError("Курс с именем ${row[1]} не найден!") }
                return@CourseCompletion
            }
            employee = repository.Employees().getByName(row[2].substringBefore(" ")) ?: run {
                _errState.update {
                    ErrorStates.ShowError(
                        "Сотрудник с именем ${row[2].substringBefore(" ")} не найден!"
                    )
                }
                return@CourseCompletion
            }
            startDate = try {
                LocalDate.parse(row[3])
            } catch (e: Exception) {
                println("Wrong date format")
                _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                return@CourseCompletion
            }
        } as T

        Routes.Departments -> Department {
            if (row[0].isNotBlank()) id = row[0].toInt()
            name = row[1]
        } as T

        Routes.Employees -> Employee {
            if (row[0].isNotBlank()) id = row[0].toInt()
            name = row[1]
            surname = row[2]
            department = repository.Departments().getByName(row[3]) ?: run {
                _errState.update { ErrorStates.ShowError("Отдел с именем ${row[3]} не найден!") }
                return@Employee
            }
            position = repository.Positions().getByName(row[4]) ?: run {
                _errState.update { ErrorStates.ShowError("Должность с именем ${row[4]} не найдена!") }
                return@Employee
            }
            hireDate = try {
                LocalDate.parse(row[5]).also {
                    if (it > LocalDate.now()) {
                        _errState.update {
                            ErrorStates.ShowError(
                                "Дата найма должна быть раньше текущей даты"
                            )
                        }
                        return@Employee
                    }
                }
            } catch (e: Exception) {
                _errState.update { ErrorStates.ShowError("Неправильный формат даты") }
                return@Employee
            }
        } as T

        Routes.Positions -> Position {
            if (row[0].isNotBlank()) id = row[0].toInt()
            name = row[1]
        } as T
    }
}