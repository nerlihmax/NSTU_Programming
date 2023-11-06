package ru.kheynov.cinemabooking.domain.useCases

import org.koin.core.component.KoinComponent
import org.koin.core.component.inject
import ru.kheynov.cinemabooking.domain.repositories.UsersRepository

class DeleteUserUseCase : KoinComponent {
    private val usersRepository: UsersRepository by inject()

    sealed interface Result {
        data object Successful : Result
        data object Failed : Result
        data object UserNotExists : Result
    }

    suspend operator fun invoke(userId: String): Result {
        if (usersRepository.getUserByID(userId) == null) return Result.UserNotExists
        return if (usersRepository.deleteUserByID(userId)) Result.Successful else Result.Failed
    }
}