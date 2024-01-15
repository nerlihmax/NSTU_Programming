package presentation

import core.TableData
import data.Repository
import data.entities.Hotel
import data.entities.Reservation
import data.entities.Room
import data.entities.User
import data.entities.asTableData
import domain.entities.asTableData
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.ktorm.database.Database
import presentation.state_holders.ErrorStates
import presentation.state_holders.Routes
import presentation.state_holders.State
import java.time.LocalDate
import java.util.*

class ViewModel {

    private var database: Database = Database.connect(
        url = "jdbc:postgresql://194.35.116.110:5433/booking",
        driver = "org.postgresql.Driver",
        user = "postgres",
        password = "DS0GTEeXHLJGX2Wt",
    )

    private val repository = Repository(database)

    private val _route: MutableStateFlow<Routes> = MutableStateFlow(Routes.Hotels)
    val route: StateFlow<Routes> = _route.asStateFlow()

    private val _state: MutableStateFlow<State> = MutableStateFlow(State.Loading)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _errState: MutableStateFlow<ErrorStates> = MutableStateFlow(ErrorStates.Idle)
    val errState: StateFlow<ErrorStates> = _errState.asStateFlow()

    private val _data: MutableStateFlow<TableData> = MutableStateFlow(TableData(emptyList(), emptyList()))
    val data: StateFlow<TableData> = _data.asStateFlow()

    private val _hotels: MutableStateFlow<List<Hotel>> = MutableStateFlow(emptyList())
    val hotels: StateFlow<List<Hotel>> = _hotels.asStateFlow()

    init {
        loadData(this._route.value)
        fetchHotels()
    }

    private fun loadData(route: Routes) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route) {
                    Routes.Hotels -> _data.update { repository.HotelsRepository().getAll().asTableData }
                    Routes.Reservations -> _data.update { repository.ReservationsRespository().getAll().asTableData }
                    Routes.Users -> _data.update { repository.UsersRepository().getAll().asTableData }
                    is Routes.Rooms -> {
                        fetchHotels()
                        _data.update {
                            repository.RoomsRepository().getAllInHotel(route.hotelId).asTableData
                        }
                    }
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

    private fun fetchHotels() {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                _hotels.update { repository.HotelsRepository().getAll() }
            }
            awaitAll(job)
            _state.update { State.Idle }
        }
    }

    fun startEditing(id: String) = _state.update { if (it is State.Editing) State.Idle else State.Editing(id) }

    fun startAdding() = _state.update { if (it is State.Adding) State.Idle else State.Adding }

    fun clearState() {
        _state.update { State.Idle }
        _errState.update { ErrorStates.Idle }
    }

    private fun showError(text: String) {
        _errState.update { ErrorStates.ShowError(text) }
    }

    fun add(row: List<String>) {
        CoroutineScope(SupervisorJob() + Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Hotels -> if (!repository.HotelsRepository().create(
                            updateData(_route.value, row) ?: return@async
                        )
                    ) {
                        _errState.update { ErrorStates.ShowError("Ошибка!") }
                        return@async
                    }

                    Routes.Reservations -> if (!repository.ReservationsRespository().create(
                            updateData(_route.value, row) ?: return@async
                        )
                    ) {
                        _errState.update { ErrorStates.ShowError("Ошибка!") }
                        return@async
                    }

                    Routes.Users -> if (!repository.UsersRepository()
                            .create(updateData(_route.value, row) ?: return@async)
                    ) {
                        _errState.update { ErrorStates.ShowError("Ошибка!") }
                        return@async
                    }

                    is Routes.Rooms -> if (!repository.RoomsRepository().create(
                            updateData(_route.value, row) ?: return@async
                        )
                    ) {
                        _errState.update { ErrorStates.ShowError("Ошибка!") }
                        return@async
                    }
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
            val job = CoroutineScope(SupervisorJob() + Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Hotels -> repository.HotelsRepository().update(updateData(_route.value, row) ?: return@async)
                    Routes.Reservations -> repository.ReservationsRespository()
                        .update(updateData(_route.value, row) ?: return@async)

                    Routes.Users -> repository.UsersRepository().update(updateData(_route.value, row) ?: return@async)
                    is Routes.Rooms -> repository.RoomsRepository()
                        .update(updateData(_route.value, row) ?: return@async)
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    fun deleteRow(id: String) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Hotels -> repository.HotelsRepository().delete(id.toIntOrNull() ?: run {
                        _errState.update { ErrorStates.ShowError("Неправильный формат id") }
                        return@async
                    })

                    Routes.Reservations -> repository.ReservationsRespository().delete(id)
                    Routes.Users -> repository.UsersRepository().delete(id)
                    is Routes.Rooms -> repository.RoomsRepository().delete(id)
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    private fun checkRooms(id: Int, roomNumber: Int): Boolean =
        repository.RoomsRepository().getAllInHotel(id).find { it.number == roomNumber } == null


//    fun showCurrentCourses() {
//        val courses = repository.getEmployeesCurrentCourses(LocalDate.now().monthValue, LocalDate.now().year)
//        _state.update { State.ShowCurrentCourses(courses) }
//    }
//
//    fun showDepartmentCourses(id: Int) {
//        val courses = repository.getPassedEmployeeCoursesByDepartment(id)
//        _state.update { State.ShowPassedCourses(courses) }
//    }
//
//    fun showPlannedCourses() {
//        val courses = repository.getEmployeesPlannedCourses()
//        _state.update { State.ShowPlannedCourses(courses) }
//    }

    fun hideInfo() = _state.update { State.Idle }

    @Suppress("UNCHECKED_CAST")
    private fun <T> updateData(route: Routes, row: List<String>): T? =
        when (route) {
            Routes.Hotels -> Hotel {
                id = if (row[0].isNotBlank()) try {
                    row[0].toInt()
                } catch (e: Exception) {
                    showError("Неправильный формат id")
                    return null
                }
                else hotels.value.last().id + 1
                name = row[1].ifBlank { showError("Название не может быть пустым"); return null }
                city = row[2].ifBlank { showError("Город не может быть пустым"); return null }
                address = row[3].ifBlank { showError("Адрес не может быть пустым"); return null }
                rating = try {
                    row[4].toInt().also {
                        if (it !in 1..5) {
                            showError("Рейтинг должен быть в диапазоне от 1 до 5")
                            return null
                        }
                    }
                } catch (e: Exception) {
                    showError("Неправильный формат рейтинга")
                    return null
                }
            } as T

            Routes.Reservations -> Reservation {
                id = row[0].ifBlank { UUID.randomUUID().toString().substring(0, 8) }
                guest = repository.UsersRepository().getById(row[1]) ?: run {
                    showError("Пользователь с ID ${row[1]} не найден!")
                    return null
                }
                room = repository.RoomsRepository().getById(row[2]) ?: run {
                    showError("Комната с ID ${row[3]} не найдена!")
                    return null
                }
                arrivalDate = try {
                    LocalDate.parse(row[3])
                } catch (e: Exception) {
                    showError("Неправильный формат даты")
                    return null
                }
                departureDate = try {
                    LocalDate.parse(row[4])
                } catch (e: Exception) {
                    showError("Неправильный формат даты")
                    return null
                }
            } as T

            Routes.Users -> User {
                userId = row[0].ifBlank { UUID.randomUUID().toString().substring(0, 8) }
                name = row[1].ifBlank { showError("Имя не может быть пустым"); return null }
            } as T

            is Routes.Rooms -> Room {
                id = row[0].ifBlank { UUID.randomUUID().toString().substring(0, 8) }
                type = row[1].ifBlank { showError("Тип не может быть пустым"); return null }
                price = try {
                    row[2].toInt()
                } catch (e: Exception) {
                    showError("Неправильный формат цены")
                    return null
                }
                number = try {
                    row[3].toInt().also {
                        if (it < 0) {
                            showError("Номер комнаты не может быть отрицательным")
                            return null
                        }
                        if (!checkRooms(route.hotelId, it)) {
                            showError("Такая комната уже существует")
                            return null
                        }
                    }
                } catch (e: Exception) {
                    showError("Неправильный формат номера")
                    return null
                }
                hotel = repository.HotelsRepository().getById(route.hotelId) ?: run {
                    showError("Отель с ID ${route.hotelId} не найден!")
                    return null
                }
            } as T
        }

    fun showRoomBooking(roomId: String) {
        _state.update { State.ShowRoomBooking(roomId) }
    }
}
