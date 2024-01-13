package ru.kheynov.hotel.domain.useCases.users

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.hotel.domain.repository.UsersRepository
import ru.kheynov.hotel.shared.domain.entities.User

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
        val user = usersRepository.getUserInfoByID(userId) ?: return Result.UserNotFound
        return Result.Successful(User(user.id, user.name, user.email))
    }
}