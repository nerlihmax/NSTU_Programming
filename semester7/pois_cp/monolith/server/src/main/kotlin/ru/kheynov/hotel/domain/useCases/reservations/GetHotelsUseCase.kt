package ru.kheynov.hotel.domain.useCases.reservations

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.shared.domain.entities.Hotel
import ru.kheynov.hotel.shared.domain.repository.ReservationsRepository

class GetHotelsUseCase : KoinComponent {
    private val reservationsRepository: ReservationsRepository by inject()

    sealed interface Result {
        data class Successful(val data: List<Hotel>) : Result
        data object Failed : Result
        data object Empty : Result
    }

    suspend operator fun invoke(): Result {
        val hotels = reservationsRepository.getHotels()
        return try {
            if (hotels.isEmpty()) Result.Empty
            else Result.Successful(hotels)
        } catch (e: Exception) {
            Result.Failed
        }
    }
}