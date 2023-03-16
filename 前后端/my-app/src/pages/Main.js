import React, {Component} from 'react';
import {Form,Input,Button} from "antd"
import requests from "../utils/requests";
class Main extends Component {
    state={
        article:""
    }
    render() {
        return (
            <div style={{
                margin:40
            }}>
                <Form
                    style={{
                        position:"relative"
                    }}
                    name="basic"
                    labelCol={{
                        span: 10,
                    }}
                    wrapperCol={{
                        span: 8,
                    }}
                    onFinish={this.onFinish}
                >
                    <Form.Item
                        label="请输入你要生成藏头诗的字符串(只能包含汉字)"
                        name="key"
                        rules={[
                            {
                                required: true,
                                message: '表单不能为空！,请输入！',
                            },
                        ]}
                    >
                        <Input />
                    </Form.Item>
                    <Form.Item
                        wrapperCol={{
                            offset: 8,
                            span: 8,
                        }}
                    >
                        <Button type="primary" htmlType="submit">
                            提交
                        </Button>
                    </Form.Item>
                </Form>
                <div style={{fontSize:"30px"}}>
                    <div>生成的藏头诗为:</div>
                    <div>{this.state.article}</div>
                </div>
                <div style={{
                    fontSize:"6px",
                position:"absolute",
                bottom:0,
                    margin:"auto",
                    left:0,
                    right:0,
                }}>
                    powered by 刘博严，王中华，李俊和，徐子超
                </div>
            </div>
        );
    }
     onFinish=async (values)=>{
         console.log(values)
        const res=await requests({
            method:"post",
            url:"http://175.178.81.93:5000",
            data:values
        })
         console.log(res)
         this.setState({
             article:res.data.poem
         })
    }
}

export default Main;