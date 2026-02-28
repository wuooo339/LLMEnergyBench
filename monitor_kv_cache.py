#!/usr/bin/env python3
"""
实时监控 vLLM KV Cache 占用情况
每秒更新一次，显示动态统计信息，并保存 trace 数据
"""

import time
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加 benchmarks 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'benchmarks'))

from util.kv_cache_monitor import KVCacheMonitor


def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')


def format_percentage(value):
    """格式化百分比"""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def format_number(value):
    """格式化数字（添加千位分隔符）"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def print_header():
    """打印表头"""
    print("=" * 80)
    print(" " * 20 + "vLLM KV Cache 实时监控")
    print("=" * 80)
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)


def print_static_info(static_config):
    """打印静态配置信息"""
    print("\n📋 静态配置信息:")
    print(f"  总 GPU Blocks:     {format_number(static_config.get('total_gpu_blocks'))}")
    print(f"  Block 大小:        {format_number(static_config.get('block_size'))} tokens/block")
    print(f"  每 Block Token 数: {format_number(static_config.get('tokens_per_block'))}")
    
    total_capacity = static_config.get('total_capacity_tokens')
    if total_capacity:
        print(f"  总容量:            {format_number(total_capacity)} tokens ({total_capacity/1024:.1f}K)")


def print_dynamic_stats(current_stats, history_stats):
    """打印动态统计信息"""
    print("\n📊 当前状态 (实时):")
    
    # 当前值
    cache_usage = current_stats.get('cache_usage_perc', 0)
    used_blocks = current_stats.get('used_blocks', 0)
    free_blocks = current_stats.get('free_blocks', 0)
    used_tokens = current_stats.get('used_tokens', 0)
    running = current_stats.get('num_requests_running', 0)
    waiting = current_stats.get('num_requests_waiting', 0)
    page_eviction_ops_total = current_stats.get('page_eviction_ops_total', 0)
    page_eviction_blocks_total = current_stats.get('page_eviction_blocks_total', 0)
    spec_draft_tokens_total = current_stats.get('spec_decode_draft_tokens_total', 0)
    spec_accepted_tokens_total = current_stats.get('spec_decode_accepted_tokens_total', 0)
    spec_acceptance_rate = current_stats.get('spec_decode_acceptance_rate', 0)
    
    print(f"  Cache 使用率:      {format_percentage(cache_usage * 100)}")
    print(f"  已用 Blocks:       {format_number(used_blocks)}")
    print(f"  空闲 Blocks:       {format_number(free_blocks)}")
    print(f"  已用 Tokens:       {format_number(used_tokens)}")
    print(f"  运行中的请求:      {format_number(running)}")
    print(f"  等待中的请求:      {format_number(waiting)}")
    print(f"  PageEviction 次数: {format_number(page_eviction_ops_total)}")
    print(f"  驱逐 Block 总数:   {format_number(page_eviction_blocks_total)}")
    print(f"  Spec Draft Tokens: {format_number(spec_draft_tokens_total)}")
    print(f"  Spec Accept Tokens:{format_number(spec_accepted_tokens_total)}")
    print(f"  Spec 接受率:       {spec_acceptance_rate:.4f}")
    
    # 历史统计（如果有）
    if history_stats and len(history_stats) > 0:
        print("\n📈 历史统计 (最近采样):")
        
        # 计算平均值
        avg_cache = sum(s.get('cache_usage_perc', 0) for s in history_stats) / len(history_stats)
        avg_used_blocks = sum(s.get('used_blocks', 0) for s in history_stats) / len(history_stats)
        avg_running = sum(s.get('num_requests_running', 0) for s in history_stats) / len(history_stats)
        avg_waiting = sum(s.get('num_requests_waiting', 0) for s in history_stats) / len(history_stats)
        evict_ops_delta = (
            history_stats[-1].get('page_eviction_ops_total', 0)
            - history_stats[0].get('page_eviction_ops_total', 0)
        )
        evict_blocks_delta = (
            history_stats[-1].get('page_eviction_blocks_total', 0)
            - history_stats[0].get('page_eviction_blocks_total', 0)
        )
        spec_draft_delta = (
            history_stats[-1].get('spec_decode_draft_tokens_total', 0)
            - history_stats[0].get('spec_decode_draft_tokens_total', 0)
        )
        spec_accepted_delta = (
            history_stats[-1].get('spec_decode_accepted_tokens_total', 0)
            - history_stats[0].get('spec_decode_accepted_tokens_total', 0)
        )
        spec_acceptance = (
            spec_accepted_delta / spec_draft_delta if spec_draft_delta > 0 else 0
        )
        
        # 计算峰值
        max_cache = max(s.get('cache_usage_perc', 0) for s in history_stats)
        max_used_blocks = max(s.get('used_blocks', 0) for s in history_stats)
        max_running = max(s.get('num_requests_running', 0) for s in history_stats)
        
        print(f"  平均 Cache 使用率: {format_percentage(avg_cache * 100)}")
        print(f"  峰值 Cache 使用率: {format_percentage(max_cache * 100)}")
        print(f"  平均已用 Blocks:   {format_number(avg_used_blocks)}")
        print(f"  峰值已用 Blocks:   {format_number(max_used_blocks)}")
        print(f"  平均运行请求数:    {format_number(avg_running)}")
        print(f"  峰值运行请求数:    {format_number(max_running)}")
        print(f"  平均等待请求数:    {format_number(avg_waiting)}")
        print(f"  PageEviction 增量: {format_number(evict_ops_delta)} ops")
        print(f"  驱逐 Block 增量:   {format_number(evict_blocks_delta)} blocks")
        print(f"  Spec Draft 增量:   {format_number(spec_draft_delta)}")
        print(f"  Spec 接受 增量:    {format_number(spec_accepted_delta)}")
        print(f"  Spec 接受率增量:   {spec_acceptance:.4f}")
        print(f"  采样数量:          {len(history_stats)}")


def print_visual_bar(label, value, max_value=100, width=50):
    """打印可视化进度条"""
    if max_value <= 0:
        percentage = 0
    else:
        percentage = min(100, (value / max_value) * 100)
    
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    print(f"  {label}: [{bar}] {percentage:.1f}%")


def print_visual_stats(current_stats):
    """打印可视化统计"""
    print("\n📉 可视化:")
    
    cache_usage_perc = current_stats.get('cache_usage_perc', 0) * 100
    print_visual_bar("Cache 使用率", cache_usage_perc, 100, 50)
    
    running = current_stats.get('num_requests_running', 0)
    print_visual_bar("运行请求数  ", running, 200, 50)  # 假设最大 200 并发
    
    waiting = current_stats.get('num_requests_waiting', 0)
    print_visual_bar("等待请求数  ", waiting, 100, 50)  # 假设最大 100 等待


def print_footer():
    """打印页脚"""
    print("\n" + "-" * 80)
    print("按 Ctrl+C 退出监控")
    print("=" * 80)


def save_trace_data(trace_data, output_dir, session_name):
    """保存 trace 数据到 JSON 文件"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = f"kv_cache_trace_{session_name}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Trace 数据已保存到: {filepath}")
    return filepath


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实时监控 vLLM KV Cache 占用')
    parser.add_argument('--host', type=str, default='localhost',
                        help='vLLM 服务器地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM 服务器端口 (默认: 8000)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='更新间隔（秒）(默认: 1.0)')
    parser.add_argument('--history-size', type=int, default=60,
                        help='保留历史记录数量 (默认: 60)')
    parser.add_argument('--output-dir', type=str, default='./kv_cache_traces',
                        help='trace 数据输出目录 (默认: ./kv_cache_traces)')
    parser.add_argument('--session-name', type=str, default='monitor',
                        help='会话名称，用于文件命名 (默认: monitor)')
    args = parser.parse_args()
    
    # 创建监控器
    base_url = f"http://{args.host}:{args.port}"
    print(f"正在连接到 vLLM 服务器: {base_url}")
    print(f"更新间隔: {args.interval} 秒")
    print(f"Trace 输出目录: {args.output_dir}")
    print(f"会话名称: {args.session_name}")
    print(f"启动监控...\n")
    
    monitor = KVCacheMonitor(
        base_url=base_url,
        interval=args.interval
    )
    
    # 启动监控
    monitor.start()
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now()
    
    # 历史数据缓存（用于实时显示）
    history_cache = {
        'cache_usage_perc': [],
        'used_blocks': [],
        'free_blocks': [],
        'used_tokens': [],
        'num_requests_running': [],
        'num_requests_waiting': [],
        'page_eviction_ops_total': [],
        'page_eviction_blocks_total': [],
        'spec_decode_draft_tokens_total': [],
        'spec_decode_accepted_tokens_total': [],
        'spec_decode_acceptance_rate': [],
        'timestamps': []
    }
    
    # Trace 数据（完整记录，用于保存）
    trace_data = {
        'metadata': {
            'session_name': args.session_name,
            'start_time': start_datetime.isoformat(),
            'host': args.host,
            'port': args.port,
            'interval': args.interval,
        },
        'static_config': {},
        'trace': []
    }
    
    try:
        # 等待第一次采样
        time.sleep(args.interval * 2)
        
        # 主循环
        iteration = 0
        while True:
            clear_screen()
            
            # 获取当前状态
            current_state = monitor.get_current_state()
            
            # 获取静态配置
            static_config = monitor.static_config
            
            # 第一次获取到静态配置时保存
            if static_config and not trace_data['static_config']:
                trace_data['static_config'] = static_config.copy()
            
            # 将当前状态转换为显示格式
            current_stats = {}
            if current_state:
                current_timestamp = time.time()
                elapsed_time = current_timestamp - start_time
                
                # 从 normalized state 转换回原始值
                current_stats['cache_usage_perc'] = current_state.get('kv_cache_usage', 0)
                current_stats['num_requests_running'] = current_state.get('num_requests_running', 0)
                current_stats['num_requests_waiting'] = current_state.get('num_requests_waiting', 0)
                current_stats['page_eviction_ops_total'] = current_state.get('page_eviction_ops_total', 0)
                current_stats['page_eviction_blocks_total'] = current_state.get('page_eviction_blocks_total', 0)
                current_stats['spec_decode_draft_tokens_total'] = current_state.get('spec_decode_draft_tokens_total', 0)
                current_stats['spec_decode_accepted_tokens_total'] = current_state.get('spec_decode_accepted_tokens_total', 0)
                current_stats['spec_decode_acceptance_rate'] = current_state.get('spec_decode_acceptance_rate', 0)
                
                # 计算 blocks 信息
                if 'kv_blocks_used_ratio' in current_state and 'total_gpu_blocks' in static_config:
                    total_blocks = static_config['total_gpu_blocks']
                    used_blocks = int(current_state['kv_blocks_used_ratio'] * total_blocks)
                    current_stats['used_blocks'] = used_blocks
                    current_stats['free_blocks'] = total_blocks - used_blocks
                    
                    if 'tokens_per_block' in static_config:
                        current_stats['used_tokens'] = used_blocks * static_config['tokens_per_block']
                
                # 记录到历史缓存（用于实时显示）
                history_cache['cache_usage_perc'].append(current_stats.get('cache_usage_perc', 0))
                history_cache['used_blocks'].append(current_stats.get('used_blocks', 0))
                history_cache['free_blocks'].append(current_stats.get('free_blocks', 0))
                history_cache['used_tokens'].append(current_stats.get('used_tokens', 0))
                history_cache['num_requests_running'].append(current_stats.get('num_requests_running', 0))
                history_cache['num_requests_waiting'].append(current_stats.get('num_requests_waiting', 0))
                history_cache['page_eviction_ops_total'].append(current_stats.get('page_eviction_ops_total', 0))
                history_cache['page_eviction_blocks_total'].append(current_stats.get('page_eviction_blocks_total', 0))
                history_cache['spec_decode_draft_tokens_total'].append(current_stats.get('spec_decode_draft_tokens_total', 0))
                history_cache['spec_decode_accepted_tokens_total'].append(current_stats.get('spec_decode_accepted_tokens_total', 0))
                history_cache['spec_decode_acceptance_rate'].append(current_stats.get('spec_decode_acceptance_rate', 0))
                history_cache['timestamps'].append(current_timestamp)
                
                # 记录到 trace 数据（用于保存）
                trace_point = {
                    'timestamp': current_timestamp,
                    'elapsed_seconds': round(elapsed_time, 2),
                    'datetime': datetime.fromtimestamp(current_timestamp).isoformat(),
                    'cache_usage_perc': round(current_stats.get('cache_usage_perc', 0) * 100, 2),
                    'used_blocks': current_stats.get('used_blocks', 0),
                    'free_blocks': current_stats.get('free_blocks', 0),
                    'used_tokens': current_stats.get('used_tokens', 0),
                    'requests_running': current_stats.get('num_requests_running', 0),
                    'requests_waiting': current_stats.get('num_requests_waiting', 0),
                    'page_eviction_ops_total': current_stats.get('page_eviction_ops_total', 0),
                    'page_eviction_blocks_total': current_stats.get('page_eviction_blocks_total', 0),
                    'spec_decode_draft_tokens_total': current_stats.get('spec_decode_draft_tokens_total', 0),
                    'spec_decode_accepted_tokens_total': current_stats.get('spec_decode_accepted_tokens_total', 0),
                    'spec_decode_acceptance_rate': round(current_stats.get('spec_decode_acceptance_rate', 0), 6),
                }
                trace_data['trace'].append(trace_point)
                
                # 只保留最近的 N 个记录
                for key in history_cache:
                    if len(history_cache[key]) > args.history_size:
                        history_cache[key] = history_cache[key][-args.history_size:]
            
            # 构建历史统计
            history_stats = []
            if len(history_cache['cache_usage_perc']) > 0:
                for i in range(len(history_cache['cache_usage_perc'])):
                    stat_point = {
                        'cache_usage_perc': history_cache['cache_usage_perc'][i],
                        'used_blocks': history_cache['used_blocks'][i],
                        'num_requests_running': history_cache['num_requests_running'][i],
                        'num_requests_waiting': history_cache['num_requests_waiting'][i],
                        'page_eviction_ops_total': history_cache['page_eviction_ops_total'][i],
                        'page_eviction_blocks_total': history_cache['page_eviction_blocks_total'][i],
                        'spec_decode_draft_tokens_total': history_cache['spec_decode_draft_tokens_total'][i],
                        'spec_decode_accepted_tokens_total': history_cache['spec_decode_accepted_tokens_total'][i],
                        'spec_decode_acceptance_rate': history_cache['spec_decode_acceptance_rate'][i],
                    }
                    history_stats.append(stat_point)
            
            # 打印信息
            print_header()
            print(f"迭代次数: {iteration + 1}")
            print(f"显示缓存: {len(history_stats)} 个采样点 (最近 {args.history_size} 个)")
            print(f"Trace 数据: {len(trace_data['trace'])} 个采样点 (完整记录)")
            elapsed = time.time() - start_time
            print(f"运行时长: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
            print_static_info(static_config)
            print_dynamic_stats(current_stats, history_stats)
            print_visual_stats(current_stats)
            print_footer()
            
            # 等待下一次更新
            time.sleep(args.interval)
            iteration += 1
            
    except KeyboardInterrupt:
        print("\n\n正在停止监控...")
        monitor.stop()
        print("监控已停止")
        
        # 记录结束时间
        end_time = time.time()
        end_datetime = datetime.now()
        duration = end_time - start_time
        
        # 更新 metadata
        trace_data['metadata']['end_time'] = end_datetime.isoformat()
        trace_data['metadata']['duration_seconds'] = round(duration, 2)
        trace_data['metadata']['total_samples'] = len(trace_data['trace'])
        trace_data['metadata']['successful_fetches'] = monitor.successful_fetches
        trace_data['metadata']['fetch_errors'] = monitor.fetch_errors
        
        # 打印最终统计
        print("\n" + "=" * 80)
        print("最终统计信息:")
        print("=" * 80)
        print(f"监控时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        print(f"总采样次数: {len(trace_data['trace'])}")
        print(f"成功采集: {monitor.successful_fetches}")
        print(f"采集错误: {monitor.fetch_errors}")
        
        if len(history_cache['cache_usage_perc']) > 0:
            # 计算统计数据
            avg_cache = sum(history_cache['cache_usage_perc']) / len(history_cache['cache_usage_perc'])
            avg_blocks = sum(history_cache['used_blocks']) / len(history_cache['used_blocks'])
            avg_running = sum(history_cache['num_requests_running']) / len(history_cache['num_requests_running'])
            avg_waiting = sum(history_cache['num_requests_waiting']) / len(history_cache['num_requests_waiting'])
            avg_spec_accept = sum(history_cache['spec_decode_acceptance_rate']) / len(
                history_cache['spec_decode_acceptance_rate']
            )
            
            max_cache = max(history_cache['cache_usage_perc'])
            max_blocks = max(history_cache['used_blocks'])
            max_running = max(history_cache['num_requests_running'])
            max_waiting = max(history_cache['num_requests_waiting'])
            total_page_evict_ops = (
                history_cache['page_eviction_ops_total'][-1]
                - history_cache['page_eviction_ops_total'][0]
            )
            total_page_evict_blocks = (
                history_cache['page_eviction_blocks_total'][-1]
                - history_cache['page_eviction_blocks_total'][0]
            )
            total_spec_draft = (
                history_cache['spec_decode_draft_tokens_total'][-1]
                - history_cache['spec_decode_draft_tokens_total'][0]
            )
            total_spec_accept = (
                history_cache['spec_decode_accepted_tokens_total'][-1]
                - history_cache['spec_decode_accepted_tokens_total'][0]
            )
            total_spec_accept_rate = (
                total_spec_accept / total_spec_draft if total_spec_draft > 0 else 0
            )
            
            print("\n平均值:")
            print(f"  Cache 使用率: {format_percentage(avg_cache * 100)}")
            print(f"  已用 Blocks: {format_number(avg_blocks)}")
            print(f"  运行请求数: {format_number(avg_running)}")
            print(f"  等待请求数: {format_number(avg_waiting)}")
            print(f"  Spec 接受率: {avg_spec_accept:.4f}")

            print("\n峰值:")
            print(f"  Cache 使用率: {format_percentage(max_cache * 100)}")
            print(f"  已用 Blocks: {format_number(max_blocks)}")
            print(f"  运行请求数: {format_number(max_running)}")
            print(f"  等待请求数: {format_number(max_waiting)}")
            print(f"  PageEviction 总次数: {format_number(total_page_evict_ops)}")
            print(f"  驱逐 Block 总数: {format_number(total_page_evict_blocks)}")
            print(f"  Spec Draft 总量: {format_number(total_spec_draft)}")
            print(f"  Spec 接受总量: {format_number(total_spec_accept)}")
            print(f"  Spec 总接受率: {total_spec_accept_rate:.4f}")
            
            # 添加统计到 trace_data
            trace_data['summary'] = {
                'average': {
                    'cache_usage_perc': round(avg_cache * 100, 2),
                    'used_blocks': round(avg_blocks, 1),
                    'requests_running': round(avg_running, 1),
                    'requests_waiting': round(avg_waiting, 1),
                    'spec_decode_acceptance_rate': round(avg_spec_accept, 6),
                },
                'peak': {
                    'cache_usage_perc': round(max_cache * 100, 2),
                    'used_blocks': max_blocks,
                    'requests_running': max_running,
                    'requests_waiting': max_waiting,
                },
                'page_eviction': {
                    'total_eviction_ops': total_page_evict_ops,
                    'total_evicted_blocks': total_page_evict_blocks,
                },
                'spec_decode': {
                    'draft_tokens_total': total_spec_draft,
                    'accepted_tokens_total': total_spec_accept,
                    'acceptance_rate': round(total_spec_accept_rate, 6),
                }
            }
        
        # 保存 trace 数据
        print("\n正在保存 trace 数据...")
        save_trace_data(trace_data, args.output_dir, args.session_name)
        
        print("\n监控会话结束！")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        monitor.stop()
        
        # 即使出错也尝试保存已收集的 trace 数据
        if len(trace_data['trace']) > 0:
            print("\n尝试保存已收集的 trace 数据...")
            try:
                trace_data['metadata']['error'] = str(e)
                trace_data['metadata']['end_time'] = datetime.now().isoformat()
                save_trace_data(trace_data, args.output_dir, f"{args.session_name}_error")
            except Exception as save_error:
                print(f"保存 trace 数据失败: {save_error}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
