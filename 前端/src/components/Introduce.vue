<script setup lang="ts">
import {reactive , ref} from 'vue'
import axios from 'axios'

import { MoreFilled } from '@element-plus/icons-vue'
import type { ButtonInstance } from 'element-plus'

const ref1 = ref<ButtonInstance>()
const ref2 = ref<ButtonInstance>()

const open = ref(true)

const API_BASE_URL = 'http://8.130.128.77:8848';

const formLabelAlign = reactive({
  就诊次数_SUM: null,
  月统筹金额_MAX: null,
  ALL_SUM: null,
  月药品金额_AVG: null,
  可用账户报销金额_SUM: null
})
const responseData1 = ref()
const responseData2 = ref()
const UploadText = async () => {

  try {

    const result = await axios({
      method: 'post',
      url: `${API_BASE_URL}/upload_text`,
      data: formLabelAlign,

    });
    responseData1.value = result.data.prediction
    responseData2.value = result.data.text
    responseData2.value = "理由：" + responseData2.value
  } catch (err) {


  }


}
const formRef = ref(null);
// 提交表单函数
const submitForm = async () => {
  const isValid = await formRef.value.validate();
  if (isValid) {
    UploadText()
  } else {
  }

};



</script>
<template>


<div class="text_flex_box">

<div class="text_input_box">

  <div class="text_input_head">方法二: 文本输入预测</div>
  <el-form label-position="top" label-width="auto" :model="formLabelAlign" style="width: 600px;margin: 0 auto;"
    ref="formRef">
    <el-form-item label="就诊次数" prop="就诊次数_SUM" :rules="[
      { required: true, trigger: 'change', message: '请补充' },
      { pattern: /(^[1-9]([0-9]+)?(\.[0-9]{1,2})?$)|(^(0){1}$)|(^[0-9]\.[0-9]([0-9])?$)/, message: '请输入正确的格式,可保留两位小数' }
    ]">
      <el-input v-model="formLabelAlign.就诊次数_SUM" oninput="value=value.replace(/[^0-9.]/g,'')" />
    </el-form-item>
    <el-form-item label="月统筹金额" prop="月统筹金额_MAX" :rules="[
      { required: true, trigger: 'change', message: '请补充' },
      { pattern: /(^[1-9]([0-9]+)?(\.[0-9]{1,2})?$)|(^(0){1}$)|(^[0-9]\.[0-9]([0-9])?$)/, message: '请输入正确的格式,可保留两位小数' }
    ]">
      <el-input oninput="value=value.replace(/[^0-9.]/g,'')" v-model="formLabelAlign.月统筹金额_MAX" />
    </el-form-item>
    <el-form-item label="月药品金额" prop="ALL_SUM" :rules="[
      { required: true, trigger: 'change', message: '请补充' },
      { pattern: /(^[1-9]([0-9]+)?(\.[0-9]{1,2})?$)|(^(0){1}$)|(^[0-9]\.[0-9]([0-9])?$)/, message: '请输入正确的格式,可保留两位小数' }
    ]">
      <el-input oninput="value=value.replace(/[^0-9.]/g,'')" v-model="formLabelAlign.ALL_SUM" />
    </el-form-item>
    <el-form-item label="总金额" prop="月药品金额_AVG" :rules="[
      { required: true, trigger: 'change', message: '请补充' },
      { pattern: /(^[1-9]([0-9]+)?(\.[0-9]{1,2})?$)|(^(0){1}$)|(^[0-9]\.[0-9]([0-9])?$)/, message: '请输入正确的格式,可保留两位小数' }
    ]">
      <el-input oninput="value=value.replace(/[^0-9.]/g,'')" v-model="formLabelAlign.月药品金额_AVG" />
    </el-form-item>
    <el-form-item label="可用账户报销金额" prop="可用账户报销金额_SUM" :rules="[
      { required: true, trigger: 'change', message: '请补充' },
      { pattern: /(^[1-9]([0-9]+)?(\.[0-9]{1,2})?$)|(^(0){1}$)|(^[0-9]\.[0-9]([0-9])?$)/, message: '请输入正确的格式,可保留两位小数' }
    ]">
      <el-input oninput="value=value.replace(/[^0-9.]/g,'')" v-model="formLabelAlign.可用账户报销金额_SUM" />
    </el-form-item>
  </el-form>
  <el-form-item>
    <el-button type="primary" @click="submitForm" style="background-color: #bed0a6;border: none;margin: 0 auto;" ref="ref2">
      上传
    </el-button>
  </el-form-item>
  <div class="text_input_show">
    <div v-if="responseData1 === '该病人涉嫌医保欺诈，请管理员注意'">检测结果：{{ responseData1.slice(0, 3) }}<span
        style="color: red;">{{ responseData1.slice(3, 9) }}</span>{{ responseData1.slice(9) }}</div>
    <div v-else-if="responseData1 === '该病人医保数据正常，不涉及骗保'">检测结果：{{ responseData1 }}</div>
    <div v-else-if="responseData1 === '有91%概率认为该病人遭受医保欺诈，请管理员注意'">检测结果：有91%概率认为该病人<span style="color: red;">遭受医保欺诈</span>，请管理员注意</div>

    <div v-else style="text-align: center;">您可以通过输入上述五个数据获取预测结果</div>

    <div>{{ responseData2 }}</div>
  </div>
  <el-tour v-model="open" type="success" :mask="false">
    <el-tour-step
      :target="ref1?.$el"
      title="文本输入预测"
      description="在表单中填写相应数据"
    />
    <el-tour-step
      :target="ref2?.$el"
      title="上传"
      description="点此上传输入的数据以获取预测结果"
      placement="bottom-start"
    />
  </el-tour>
</div>

</div>
<!-- <div class="backgd" id="cont1">
        <div class="backgd_head">
          项目背景
        </div>
        <div class="backgd_content">
          <p class="cont">响应国家号召，打击欺诈骗保</p>
          <img src="../store/xi.jpeg" alt="" class="xi">
          <p class="cont_main">
            习近平总书记在十九届中央纪委四次全会讲话中强调， 要坚决查处医疗机构内外勾结欺诈骗保行为，建立和强化长效监管机制。
医疗保障基金是人民群众的“看病钱”“救命钱”。习近平总书记深刻指出：“我们建立全民医保制度的根本目的，就是要解除全体人民的疾病医疗后顾之忧。”医疗保障作为社会保障的重要内容，在增进民生福祉、维护社会和谐、推动实现共同富裕方面发挥着重要作用。
          </p>
          <p class="cont_main">
            医保基金是人民群众的“保命钱”，加强医保基金监管是首要任务。但由于保险欺诈手段多样、隐蔽性强，给识别和防范带来了极大的挑战。一方面，由于医疗服务的复杂性，加上医疗数据量巨大，很难以人力去识别是否有欺诈嫌疑，且人力成本和时间成本高；另一方面，我国医疗保险制度尚未完全成熟，很难去从根源上界定红线、清除欺诈行为。
医疗骗保智能监测识别是破解监管痛点难点问题的重要举措之一。因此，想加强医疗保险欺诈的识别和打击力度，确保医疗保险基金的安全，不仅需要政府部门加强立法和监管，提高违法成本，更需要借助技术手段，如大数据分析、人工智能、区块链等，提升医疗保险欺诈的识别效率和准确性，构建一个更加公正、高效、可持续的医疗保障体系。
          </p>
        </div>
      </div>
      <div class="gap"></div>
      <div class="backgd" id="cont2">
        <div class="backgd_head">
          项目介绍
        </div>
        <div class="backgd_content">
          <p class="cont">医保瑞盾:智能医疗保险诈骗监测系统</p>
          
          <p class="cont_main">
            项目名为《医保瑞盾:智能医疗保险诈骗监测系统》(简称:医保瑞盾系统),专为识别和预防医疗保险诈骗行为而设计。系统结合了先进的机器学习与深度学习技术,采用GAN(生成对抗网络)、孤立森林、XGBoost、LightGBM、多层感知器(MLP)和自编码器(AE)等算法，通过深入分析医疗保险数据中的异常模式，有效自动识别出潜在的诈骗案例。
          </p>
          <p class="cont_main">
            在技术实现上,医保瑞盾系统基于Python的Scikit-learn库和Keras库进行开发,创新性地将多种算法整合应用，以达到高准确率与高召回率的标准。我们的系统通过综合利用医疗记录、保险理赔数据的结构化特征和非结构化文本信息,实现对诈骗行为的精准识别。系统采用双分支策略，一方面对数据进行异常检测分类，另一方面利用深度学习模型挖掘复杂模式和隐含关系，从而有效提升了识别的准确性和效率。
          </p>
          <img src="../store/introduce.png" alt="" class="xi">
          <p class="cont_main">
            系统设计考虑到易用性和可访问性,提供了基于Web的用户界面,使得保险公司和医疗机构的工作人员能够轻松使用系统进行诈骗监测和分析。通过云计算技术，医保瑞盾系统大幅减少了用户端的硬件需求，使得用户无需进行复杂的安装和配置即可享受服务。
          </p>
          <p class="cont_main">
            综上所述，医保瑞盾系统以其先进的技术和高效的诈骗识别能力，为医疗保险行业提供了一种创新和有效的诈骗防范解决方案，显著优化了保险公司和医疗机构的风险管理能力，保障医疗基金不流失，人民利益不被侵害，为保护公众利益和维护医疗保险体系的完整性做出了贡献，有利于维护社会稳定。
          </p>
        </div>
      </div>
      <div class="backgd" id="cont1">
        <div class="backgd_head">
          项目背景
        </div>
        <div class="backgd_content">
          <p class="cont">响应国家号召，打击欺诈骗保</p>
          <img src="../store/xi.jpeg" alt="" class="xi">
          <p class="cont_main">
            习近平总书记在十九届中央纪委四次全会讲话中强调， 要坚决查处医疗机构内外勾结欺诈骗保行为，建立和强化长效监管机制。
医疗保障基金是人民群众的“看病钱”“救命钱”。习近平总书记深刻指出：“我们建立全民医保制度的根本目的，就是要解除全体人民的疾病医疗后顾之忧。”医疗保障作为社会保障的重要内容，在增进民生福祉、维护社会和谐、推动实现共同富裕方面发挥着重要作用。
          </p>
          <p class="cont_main">
            医保基金是人民群众的“保命钱”，加强医保基金监管是首要任务。但由于保险欺诈手段多样、隐蔽性强，给识别和防范带来了极大的挑战。一方面，由于医疗服务的复杂性，加上医疗数据量巨大，很难以人力去识别是否有欺诈嫌疑，且人力成本和时间成本高；另一方面，我国医疗保险制度尚未完全成熟，很难去从根源上界定红线、清除欺诈行为。
医疗骗保智能监测识别是破解监管痛点难点问题的重要举措之一。因此，想加强医疗保险欺诈的识别和打击力度，确保医疗保险基金的安全，不仅需要政府部门加强立法和监管，提高违法成本，更需要借助技术手段，如大数据分析、人工智能、区块链等，提升医疗保险欺诈的识别效率和准确性，构建一个更加公正、高效、可持续的医疗保障体系。
          </p>
        </div>
      </div> -->
</template>
<style scoped>
.nav_r_box{
  display: flex;
  flex-direction: column;
  align-content: center;
  justify-content: center;
  width: 80px;
  position: fixed;
  background-color: #f4f9f3;
  left: 50%;
  translate: 660px 265px;
  top: 0;
  font-size: 27px;
  font-weight: 500;
  color: #1f9792;

}
.nav_r_box a{
  display: flex;
  color: black;
  border: 2px solid #5c8e8a;
  margin-top:-2px ;
  align-content: center;
  justify-content: center;
  width: 80px;
  margin: 0 auto;

}
.xi{
  width: 400px;
  height: 300px;
  float: left;
  padding: 30px;
  border-radius: 40px;
}
.cont_main{
  padding: 20px  60px;
  font-size: 25px;
  text-indent: 4ch;
}
.cont{
  padding: 20px 0;
  text-indent: 2ch;
  font-size: 28px;
  font-weight: 600;
  color: #5d8f8b;
}
.backgd_content{
  padding: 20px;
  
}
.backgd_content{
  background-color: #f4f9f3;
  border-radius: 10px;
}
.backgd{
  margin: 40px auto;
  
}
.backgd_head{
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 30px;
  width: 360px;
  height: 64px;
  background-color: #74b394;
  border-radius: 32px;
  margin:20px auto;
  font-weight: 700;
  color: #f4f9f3;
}
.gap{
  width: 1300px;
  height: 5px;
  background-color: #77b396;
  margin: 30px auto;
}
.text_input_head {
  margin: 20px auto;
  display: flex;
  align-items: end;
  justify-content: center;
  font-size: 30px;
  color: #1f9792;
  font-weight: 800;
  font-family: Black;
}

.loading_box {
  width: 260px;
  height: 48px;
  padding: 20px;
  background-color: #f0f0f0;
  position: absolute;
  top: 412px;
  left: 50%;
  translate: -50% 0;
  border-radius: 5px;
  display: flex;
  justify-self: center;
  align-items: center;

}

.text_flex_box {
  display: flex;
  flex-direction: column;
  
}

.text_input_show {
  width: 800px;

  background-color: white;
  padding: 20px;
  margin: 0 auto;
  border-radius: 8px;
  line-height: 25px;
  font-size: 20px;

}

.text_input_box {
  
  width: 1200px;
  height: 615px;
  margin: 0 auto;
  padding: 0 auto;
  background-color: #f4f9f3;
  border-radius: 8px;
}

</style>
