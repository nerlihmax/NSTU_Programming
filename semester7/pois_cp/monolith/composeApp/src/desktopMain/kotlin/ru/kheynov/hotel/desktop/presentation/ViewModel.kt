package ru.kheynov.hotel.desktop.presentation

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.desktop.presentation.stateHolders.ErrorStates
import ru.kheynov.hotel.desktop.presentation.stateHolders.Routes
import ru.kheynov.hotel.desktop.presentation.stateHolders.State
import ru.kheynov.hotel.shared.data.repository.ClientReservationsRepository
import ru.kheynov.hotel.shared.data.repository.ClientUsersRepository
import ru.kheynov.hotel.shared.domain.entities.DisplayableData
import ru.kheynov.hotel.shared.domain.entities.UserMetadataState

class ViewModel : KoinComponent {

    private val usersRepository: ClientUsersRepository by inject()
    private val reservationRepository: ClientReservationsRepository by inject()

    private val _route: MutableStateFlow<Routes> = MutableStateFlow(Routes.Bookings)
    val route: StateFlow<Routes> = _route.asStateFlow()

    private val _state: MutableStateFlow<State> = MutableStateFlow(State.Loading)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _errState: MutableStateFlow<ErrorStates> = MutableStateFlow(ErrorStates.Idle)
    val errState: StateFlow<ErrorStates> = _errState.asStateFlow()

    private val _data: MutableStateFlow<List<DisplayableData>> = MutableStateFlow(emptyList())
    val data: StateFlow<List<DisplayableData>> = _data.asStateFlow()

    private val _metadata: MutableStateFlow<UserMetadataState> =
        MutableStateFlow(UserMetadataState.Loading)
    val metadata: StateFlow<UserMetadataState> = _metadata.asStateFlow()

    init {
        loadData(this._route.value)

        CoroutineScope(Dispatchers.Default).launch {
            val job = CoroutineScope(Dispatchers.IO).async {
                val user = usersRepository.getUserInfo().fold(
                    onSuccess = { it },
                    onFailure = { err ->
                        _errState.update { ErrorStates.ShowError(err.javaClass.simpleName.toString()) }
                        return@async
                    }
                ) ?: run {
                    _errState.update { ErrorStates.ShowError("User not found") }
                    return@async
                }
                val employment = usersRepository.getUserEmployment().fold(
                    onSuccess = { it },
                    onFailure = { err ->
                        _errState.update { ErrorStates.ShowError(err.javaClass.simpleName.toString()) }
                        return@async
                    }
                )
                _metadata.update {
                    UserMetadataState.UserMetadata(
                        name = user.name,
                        isEmployee = employment.isEmployee,
                        hotel = employment.hotel ?: kotlin.run {
                            _errState.update { ErrorStates.ShowError("Пользователь не является сотрудником!") }
                            return@async
                        }
                    )
                }
            }
            awaitAll(job)
        }
    }

    private fun loadData(route: Routes) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (route) {
                    Routes.Bookings -> _data.update { reservationRepository.getReservations() }
                    Routes.Rooms -> _data.update {
                        reservationRepository.getRooms(
                            (_metadata.value as? UserMetadataState.UserMetadata)?.hotel?.id ?: run {
                                _errState.update { ErrorStates.ShowError("Ошибка!") }
                                return@async
                            }
                        )
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

    fun startEditing(id: Int) =
        _state.update { if (it is State.Editing) State.Idle else State.Editing(id) }

    fun startAdding() = _state.update { if (it is State.Adding) State.Idle else State.Adding }

    fun clearErrorState() {
        _errState.update { ErrorStates.Idle }
    }

    fun add(row: List<String>) {
        CoroutineScope(Dispatchers.Default).launch {
            _state.update { State.Loading }
            val job = CoroutineScope(Dispatchers.IO).async {
                when (_route.value) {
                    Routes.Bookings -> TODO()
                    Routes.Rooms -> TODO()
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
                    Routes.Bookings -> TODO()
                    Routes.Rooms -> TODO()
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
                    Routes.Bookings -> {
                        reservationRepository.deleteReservation(id)
                    }

                    else -> Unit
                }
            }
            awaitAll(job)
            _state.update { State.Idle }
            loadData(_route.value)
        }
    }

    fun hideInfo() = _state.update { State.Idle }
}