package preprocessing;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class RatingsPreprocessor {
	private final static int LINES_PER_REVIEW = 11;		//If only MultiLineInputFormat split on multiples of this.
	private final static int REVIEWS_PER_BLOCK = 60000;	//Instead, we do this so we don't get 400,000,000 mappers.
	private final static int BLOCKS_PER_SPLIT = 3;		//This gives us a reasonable 3x64mb chunk per mapper.
	private final static int LINES_PER_SPLIT = (LINES_PER_REVIEW * REVIEWS_PER_BLOCK * BLOCKS_PER_SPLIT);
	private final static int MIN_REVIEWS = 2;	//minimum number of reviews a product must have to be recorded
	
	public static class RatingsPreprocessorMapper extends Mapper<LongWritable, Text, Text, Text> {
		String keyTerm;
		ArrayDeque<String> valueTerms = new ArrayDeque<String>();
		
		@Override
		protected void setup(Mapper<LongWritable, Text, Text, Text>.Context context)
				throws IOException, InterruptedException {			
			
			super.setup(context);
			
			keyTerm = "product/productId: ";
			valueTerms.add("review/userId: ");	//add terms in the order you want them in the output
			valueTerms.add("review/score: ");	//if you want the review text, change the split(regex) below
		}

		@Override
		public void map(LongWritable offset, Text text, Context context) throws IOException, InterruptedException {
			ArrayDeque<String> reviews = new ArrayDeque<String>();
			Collections.addAll(reviews, text.toString().split("review/text: .*\n\n"));	
			
			for(String review : reviews){				
				if(review.contains("review/userId: unknown"))	//skip
					continue;
				
				int keyStart = review.indexOf(keyTerm) + keyTerm.length();
				int keyEnd = review.indexOf("\n", keyStart);
				String key = review.substring(keyStart, keyEnd);
				
				StringBuilder values = new StringBuilder();
				boolean needComma = false;
				
				for(String valueTerm : valueTerms){
					if(needComma)
						values.append(",");
					
					int vtStart = review.indexOf(valueTerm) + valueTerm.length();
					int vtEnd = review.indexOf("\n", vtStart);
					values.append(review.substring(vtStart, vtEnd));
					needComma = true;
				}
				
				context.write(new Text(key), new Text(values.toString()));
				
			}
		}
	}
	
	public static class RatingsPreprocessorReducer extends Reducer<Text, Text, NullWritable, Text>{		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			HashSet<String> allValues = new HashSet<String>();
			NullWritable nothing = NullWritable.get();
			
			for(Text value : values){
				allValues.add(value.toString());
			}
			
			if(allValues.size() >= MIN_REVIEWS){
				for(String value : allValues)
					context.write(nothing, new Text(key.toString() + "," + value));
			}				
		}
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException, URISyntaxException {
		Configuration conf = new Configuration();
		conf.set("mapred.job.queue.name", "hadoop02");
		
		GenericOptionsParser gop = new GenericOptionsParser(conf, args);
		String[] otherArgs = gop.getRemainingArgs();
		Job job = Job.getInstance(conf, "ratings preprocessing");
		
		job.setJarByClass(RatingsPreprocessor.class);
		job.setMapperClass(RatingsPreprocessorMapper.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setInputFormatClass(MultiLineInputFormat.class);
		MultiLineInputFormat.setInputPaths(job, new Path(otherArgs[0]));
		MultiLineInputFormat.setNumLinesPerSplit(job, LINES_PER_SPLIT);	
		job.setReducerClass(RatingsPreprocessorReducer.class);
		job.setOutputKeyClass(NullWritable.class);
		job.setOutputValueClass(Text.class);
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		try{
			System.exit(job.waitForCompletion(true) ? 0 : 1);
		}
		catch (ClassNotFoundException e){
			e.printStackTrace();
		}
		catch (InterruptedException e){
			e.printStackTrace();
		}
	}
}