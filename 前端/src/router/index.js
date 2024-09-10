import { createRouter, createWebHashHistory } from 'vue-router'
import Home from "@/components/Home.vue"
import detail from "@/components/Detail.vue"
import Introduce from "@/components/Introduce.vue"

const router = createRouter({
  history: createWebHashHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
    },
    {
      path: '/upload_file',
      name: 'upload_file',
      component: detail,
    },
    {
      path: '/upload_text',
      name: 'upload_text',
      component: Introduce,
    }
    
  ]
})

export default router
