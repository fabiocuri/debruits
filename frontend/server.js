const express = require('express')
const app = express()
const bodyParser = require('body-parser')

app.use(bodyParser.json())
// Whenever we open localhost:3000 on the browser, we get as a return the index.html that is hosted in the path below.
app.use('/', express.static('./dist/'))

app.listen(3100, function () {
    console.log("app listening on port 3100")
})