package ru.kheynov.hotel.domain.useCases

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.domain.entities.User
import ru.kheynov.hotel.domain.repository.UsersRepository

class GetUserDetailsUseCase : KoinComponent {
    private val usersRepository: UsersRepository by inject()

    sealed interface Result {
        data class Successful(val user: User) : Result
        data object Failed : Result
        data object UserNotFound : Result
    }

    suspend operator fun invoke(
        userId: String,
    ): Result {
        val user = usersRepository.getUserByID(userId) ?: return Result.UserNotFound
        return Result.Successful(User(user.id, user.name, user.email))
    }
}