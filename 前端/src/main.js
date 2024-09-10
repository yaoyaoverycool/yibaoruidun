import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import ElTableInfiniteScroll from "el-table-infinite-scroll";
const app = createApp(App)
app.use(router)
app.use(ElementPlus)
app.use(ElTableInfiniteScroll)
app.mount('#app')