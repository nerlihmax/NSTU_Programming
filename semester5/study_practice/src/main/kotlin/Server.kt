import java.net.ServerSocket


fun main() {
    val server = ServerSocket(8080)
    while (true) {
        val socket = server.accept()
        val writer = socket.getOutputStream().writer()
        val reader = socket.getInputStream().reader()
        writer.write("Hello, world!")
        writer.flush()
        println(reader.readText())
        socket.close()
    }
}
