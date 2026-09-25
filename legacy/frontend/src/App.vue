<script>
import Header from './Header.vue'
import image from "./fernandona.png"
export default {
    name: 'App',
    components: {
        Header
    },
    data() {
        return {
            image: image
        }
    }
}
</script>

<template>
    <div class="container-one">
        <Header title="De Bruits" action="click" color="blue" bg="yellow"/>
        <img :src="image">
    </div>
    <div class="container-two">
        <Header title="De Bruits 2" action="click 2" color="black" bg="green"/>
    </div>
</template>

<style>

#app {
    text-align: center;
    margin-top: 60px
}

.container-one {
    border: 1px solid black;
    padding: 30px;
    max-width: 500px;
    margin: auto;
}

.container-two {
    border: 1px solid black;
    padding: 30px;
    max-width: 500px;
    margin: auto;
}

img {
    width: 200px
}

</style>