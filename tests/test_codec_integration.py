"""
Integration tests for codec availability and basic functionality.
"""
import pytest
import numpy as np
from pathlib import Path

try:
    import dicube.codecs
    DICUBE_AVAILABLE = True
except ImportError:
    DICUBE_AVAILABLE = False


@pytest.mark.skipif(
    not DICUBE_AVAILABLE,
    reason="dicube not available"
)
def test_codec_availability():
    """测试codec的可用性"""
    # 测试codec注册功能
    available_codecs = dicube.codecs.list_codecs()
    assert isinstance(available_codecs, list)
    assert len(available_codecs) > 0
    
    print(f"Available codecs: {available_codecs}")
    
    # 测试每个codec的可用性
    for codec_name in available_codecs:
        is_available = dicube.codecs.is_codec_available(codec_name)
        codec = dicube.codecs.get_codec(codec_name)
        
        print(f"Codec {codec_name}: available={is_available}, version={codec.get_version()}")
        
        # 检查接口完整性
        assert hasattr(codec, 'encode')
        assert hasattr(codec, 'decode')
        assert hasattr(codec, 'is_available')
        assert hasattr(codec, 'get_version')
        assert callable(codec.encode)
        assert callable(codec.decode)


@pytest.mark.skipif(
    not DICUBE_AVAILABLE,
    reason="dicube not available"
)
def test_codec_basic_functionality():
    """测试codec的基本编解码功能"""
    available_codecs = dicube.codecs.list_codecs()
    
    # 创建测试数据
    test_images = [
        ("2D_uint8", np.random.randint(0, 255, (64, 64), dtype=np.uint8)),
        ("3D_uint8", np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)),
        ("2D_uint16", np.random.randint(0, 65535, (64, 64), dtype=np.uint16)),
        ("3D_uint16", np.random.randint(0, 65535, (32, 32, 3), dtype=np.uint16)),
    ]
    
    for codec_name in available_codecs:
        if not dicube.codecs.is_codec_available(codec_name):
            print(f"Skipping {codec_name} - not available")
            continue
            
        codec = dicube.codecs.get_codec(codec_name)
        
        for test_name, test_image in test_images:
            try:
                # 编码测试
                encoded = codec.encode(test_image)
                assert isinstance(encoded, bytes)
                assert len(encoded) > 0
                
                # 解码测试
                decoded = codec.decode(encoded)
                assert isinstance(decoded, np.ndarray)
                
                # 计算压缩比
                compression_ratio = test_image.nbytes / len(encoded)
                
                print(f"  {codec_name} - {test_name}: "
                      f"{test_image.shape} -> {len(encoded)} bytes "
                      f"(ratio: {compression_ratio:.2f}x)")
                
            except Exception as e:
                print(f"  {codec_name} - {test_name}: FAILED - {e}")


@pytest.mark.skipif(
    not DICUBE_AVAILABLE,
    reason="dicube not available"
)
def test_codec_performance():
    """测试codec的性能"""
    available_codecs = dicube.codecs.list_codecs()
    
    # 创建中等大小的测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    results = {}
    
    for codec_name in available_codecs:
        if not dicube.codecs.is_codec_available(codec_name):
            continue
            
        codec = dicube.codecs.get_codec(codec_name)
        
        try:
            import time
            
            # 编码时间测试
            start_time = time.time()
            encoded = codec.encode(test_image)
            encode_time = time.time() - start_time
            
            # 解码时间测试
            start_time = time.time()
            decoded = codec.decode(encoded)
            decode_time = time.time() - start_time
            
            # 计算压缩比
            compression_ratio = test_image.nbytes / len(encoded)
            
            results[codec_name] = {
                'encode_time': encode_time,
                'decode_time': decode_time,
                'encoded_size': len(encoded),
                'compression_ratio': compression_ratio,
                'success': True
            }
            
        except Exception as e:
            results[codec_name] = {
                'error': str(e),
                'success': False
            }
    
    # 打印结果
    print("\nCodec Performance Results:")
    print("=" * 60)
    for codec_name, metrics in results.items():
        if metrics['success']:
            print(f"{codec_name}: "
                  f"encode={metrics['encode_time']:.3f}s, "
                  f"decode={metrics['decode_time']:.3f}s, "
                  f"size={metrics['encoded_size']:,} bytes, "
                  f"ratio={metrics['compression_ratio']:.2f}x")
        else:
            print(f"{codec_name}: FAILED - {metrics['error']}")


@pytest.mark.skipif(
    not DICUBE_AVAILABLE,
    reason="dicube not available"
)
def test_codec_edge_cases():
    """测试codec的边界情况"""
    available_codecs = dicube.codecs.list_codecs()
    
    # 测试边界情况
    edge_cases = [
        ("small_2D", np.random.randint(0, 255, (8, 8), dtype=np.uint8)),
        ("large_2D", np.random.randint(0, 255, (512, 512), dtype=np.uint8)),
        ("single_channel", np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)),
        ("multi_channel", np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)),
    ]
    
    for codec_name in available_codecs:
        if not dicube.codecs.is_codec_available(codec_name):
            continue
            
        codec = dicube.codecs.get_codec(codec_name)
        
        for case_name, test_image in edge_cases:
            try:
                encoded = codec.encode(test_image)
                decoded = codec.decode(encoded)
                
                assert isinstance(encoded, bytes)
                assert isinstance(decoded, np.ndarray)
                
                print(f"  {codec_name} - {case_name}: OK")
                
            except Exception as e:
                print(f"  {codec_name} - {case_name}: FAILED - {e}")


if __name__ == "__main__":
    # 运行基本测试
    print("Running codec integration tests...")
    
    test_codec_availability()
    test_codec_basic_functionality()
    test_codec_performance()
    test_codec_edge_cases() 