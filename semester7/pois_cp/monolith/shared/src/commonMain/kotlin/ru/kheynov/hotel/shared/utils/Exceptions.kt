package ru.kheynov.hotel.shared.utils

sealed class DomainException : Exception()

class BadRequestException : DomainException()
class UnauthorizedException : DomainException()
class ServerSideException : DomainException()
class NetworkException : DomainException()
class ForbiddenException : DomainException()

class EmptyFieldException : IllegalStateException()
class UnableToPerformOperation : Exception()
